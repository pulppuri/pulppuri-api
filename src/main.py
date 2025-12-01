import json
from base64 import b64decode, b64encode
from typing import Optional
from urllib.request import urlopen

import psycopg2
from bs4 import BeautifulSoup
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

import env

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://172.20.99.51:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conn = conn = psycopg2.connect(host=env.PG_HOST,
                               port=env.PG_PORT,
                               database=env.PG_DATABASE,
                               user=env.PG_USERNAME,
                               password=env.PG_PASSWORD)
model = SentenceTransformer("BAAI/bge-m3")
client = genai.Client(api_key=env.GEMINI_API_KEY)


def parse_token(x: str) -> Optional[dict[str, str]]:
    body: dict[str, str] = json.loads(b64decode(x.encode()).decode())

    if not isinstance(body["uid"], int):
        return None
    else:
        return body


@app.get("/")
async def echo():
    return {"Hello": "World"}

@app.get("/regions")
async def list_regions(q: str, page: int = 1):
    with conn.cursor() as cur:
        PAGE_SIZE = 10
        query = f"%{q}%"
        offset = (page - 1) * PAGE_SIZE if page > 0 else 0
        cur.execute("""SELECT id, full_name, display_name
                    FROM regions
                    WHERE full_name LIKE %s OR aux LIKE %s
                    ORDER BY id
                    OFFSET %s ROWS
                    FETCH NEXT %s ROWS ONLY;""",
                    (query, query, offset, PAGE_SIZE))
        result = [ { "id": id, "full_name": fn, "display_name": dn }
                   for id, fn, dn in cur.fetchall() ]

    return result

class UserDto(BaseModel):
    age: int
    job: str
    rid: int
    gender: str
    nickname: str

@app.post("/users")
async def create_user(user_dto: UserDto):
    def create_token(uid: int):
        token_str = json.dumps({ "uid": uid })

        return b64encode(token_str.encode()).decode()

    with conn.cursor() as cur:
        cur.execute("""INSERT INTO users (age, job, rid, gender, nickname)
                    SELECT %s, %s, %s, %s, %s
                    WHERE EXISTS (SELECT * FROM regions WHERE id=%s) RETURNING id;""",
                    (user_dto.age, user_dto.job, user_dto.rid, user_dto.gender, user_dto.nickname, user_dto.rid))
        result = cur.fetchone()
        if result:
            uid = result[0]
        else:
            raise HTTPException(400, "No such rid")

    conn.commit()

    return { "token": create_token(uid) }

class ProbDefDto(BaseModel):
    title: str
    rid: int
    categories: list[str]
    problem: str

class GuideDto(BaseModel):
    guide_1: str = Field(description="guideline 1")
    guide_2: str = Field(description="guideline 2")
    guide_3: str = Field(description="guideline 3")
    guide_4: str = Field(description="guideline 4")

@app.post("/guidelines")
def list_guidelines(prob_def: ProbDefDto, authorization: Optional[str] = Header(default=None)):
    if not authorization:
        raise HTTPException(401, "Authorization header not found")
    elif not (token:=parse_token(authorization)):
        raise HTTPException(403, "Authorization failed")

    text = "\n".join([
        "[질의]",
        f"제목: {prob_def.title}",
        f"유형: {', '.join(prob_def.categories)}",
        f"문제 상황: {prob_def.problem}"
    ])
    vec = model.encode(text)
    vec_str = f"[{','.join(map(str, vec))}]"

    with conn.cursor() as cur:
        cur.execute("""SELECT examples.id, title, display_name, STRING_AGG(name, ',' ORDER BY name) AS categories, 1 - (vec <=> %s) AS sim
                    FROM regions, examples, tags, ex_tags
                    WHERE regions.id=examples.rid AND examples.id=ex_tags.eid AND tags.id=ex_tags.tid
                    GROUP BY (examples.id, display_name)
                    ORDER BY vec <=> %s LIMIT 8;""",
                    (vec_str, vec_str))
        results = [ { "id": id, "title": title, "region": region, "categories": cat.split(","), "sim": sim }
                   for id, title, region, cat, sim in cur.fetchall() ]

    prompt = f"""당신은 행정 정책 분석가이자 제안서 작성 전문가입니다. 사용자로부터 받은 '정책 제안 제목'과 '문제 정의'를 기반으로, 해결 방안을 구체화하고 기대 효과를 명확히 추론할 수 있도록 가장 핵심적인 3~4가지 추가 정보를 사용자에게 간결한 질문 형태로 요청해야 합니다.

    사용자가 제시한 제목과 문제 정의는 다음과 같습니다.

    제목: {prob_def.title}
    유형: {', '.join(prob_def.categories)}
    문제 정의: {prob_def.problem}

    이 정보를 바탕으로, 해당 문제의 해결책을 구체적인 정책으로 발전시키고 그 기대 효과를 현실적으로 추론하기 위해 사용자에게 반드시 알아야 할 추가 정보를 다음 [가이드라인]과 같이 질문 리스트를 JSON 형식으로 제시해 주세요. 질문은 사용자의 입력에 부합하며 구체적이고 측정 가능하며, 정책의 실현 가능성을 높이는 데 초점을 맞춰야 합니다.

    [가이드라인, 예시 상황: 공용 자전거 설치에 관한 요구]
    - 구체적인 설치 장소 및 규모: (예시 정책의 경우) 공용 자전거 거치대의 구체적인 설치 희망 장소 3곳 이상과 예상 설치 대수는 몇 대인가요? (예: 옥천역, 옥천군청, 옥천성모병원 등)
    - 핵심 타겟층 및 이용 목적: 해당 공용 자전거의 주 이용층(예: 학생, 직장인, 관광객)은 누구이며, 주로 어떤 목적(출퇴근, 장보기, 관광)으로 이용할 것으로 예상되나요?
    - 예산/유지 관리 방안: 타 지역의 유사 사업 예산 규모를 참고하여 예상되는 연간 유지보수 비용은 어느 정도이며, 관리 주체(지자체 직영 vs. 위탁)는 어떻게 하는 것이 좋을까요?
    - 기대 효과 측정 지표: 정책 시행 후 효과를 측정할 구체적인 지표(KPI)는 무엇인가요? (예: 대중교통 이용률 감소율, 자전거 이용자 수 증가율 등)"""
    guildlines = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={ "response_mime_type": "application/json",
                 "response_json_schema": GuideDto.model_json_schema() }
    ).text or "{}"

    return {
        "examples": results,
        "guidelines": json.loads(guildlines)
    }

@app.get("/proposals")
def list_proposal(q: Optional[str] = None, page: int = 1):
    PAGE_SIZE = 10
    query = f"%{q}%" if q else "%%"
    offset = (page - 1) * PAGE_SIZE if page > 0 else 0

    with conn.cursor() as cur:
        cur.execute("""SELECT proposals.id, title, display_name AS region, STRING_AGG(name, ',' ORDER BY name) AS categories, nickname
                    FROM proposals, regions, pr_tags, tags, users
                    WHERE proposals.rid=regions.id AND proposals.id=pr_tags.pid AND pr_tags.tid=tags.id AND proposals.uid=users.id AND title LIKE %s
                    GROUP BY (proposals.id, display_name, nickname)
                    ORDER BY proposals.id DESC
                    OFFSET %s ROWS
                    FETCH NEXT %s ROWS ONLY;""",
                    (query, offset, PAGE_SIZE))
        results = [ { "id": id, "title": title, "region": region, "categories": cat.split(","), "nickname": nickname }
                    for id, title, region, cat, nickname in cur.fetchall() ]

    return { "proposals": results }

class ProposalDto(BaseModel):
    rid: int
    title: str
    eid: Optional[int] = None
    problem: str
    method: str
    effect: str
    tags: list[str]

@app.post("/proposals")
def create_proposal(pro_dto: ProposalDto, authorization: Optional[str] = Header(default=None)):
    if not authorization:
        raise HTTPException(401, "Authorization header not found")
    elif not (token:=parse_token(authorization)):
        raise HTTPException(403, "Authorization failed")

    uid = token["uid"]
    with conn.cursor() as cur:
        cur.execute("""WITH pids AS (
                    INSERT INTO proposals (eid, rid, uid, title, problem, method, effect) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id)
                    INSERT INTO pr_tags (pid, tid) SELECT pids.id, tags.id FROM pids, tags WHERE name IN %s RETURNING pid;""", (pro_dto.eid, pro_dto.rid, uid, pro_dto.title, pro_dto.problem, pro_dto.method, pro_dto.effect, tuple(pro_dto.tags)))
        results = cur.fetchall()
    conn.commit()
    if len(results) == 0:
        raise HTTPException(500)
    else:
        return { "pid": results[0][0] }

@app.get("/proposals/{pid}")
def get_proposal(pid: int):
    with conn.cursor() as cur:
        cur.execute("""SELECT proposals.id, eid, title, problem, method, effect, display_name AS region, STRING_AGG(name, ',' ORDER BY name) AS categories, nickname
                    FROM proposals, regions, pr_tags, tags, users
                    WHERE proposals.rid=regions.id AND proposals.id=pr_tags.pid AND pr_tags.tid=tags.id AND proposals.uid=users.id AND proposals.id=%s
                    GROUP BY (proposals.id, display_name, nickname);""",
                    (pid, ))
        results = [ { "id": id, "eid": eid, "title": title, "problem": problem, "method": method, "effect": effect, "region": region, "categories": cat.split(","), "nickname": nickname }
                    for id, eid, title, problem, method, effect, region, cat, nickname in cur.fetchall() ]

    return { "proposal": results[0] }

class HelperDto(BaseModel):
    title: str
    problem: str
    method: str
    effect: str

class HelperResDto(BaseModel):
    problem: str
    method: str
    effect: str

@app.post("/helper")
def helper(pro_dto: HelperDto):
    prompt = f"""당신은 정부/기관 대상의 정책 제안서 전문 편집자입니다. 사용자로부터 정책 제안서의 초안을 "제목", "문제 정의", "해결 방법", "기대 효과"의 4가지 섹션으로 구분하여 전달받았습니다.
    당신의 목표는 이 초안의 내용을 해치지 않으면서 설득력, 명확성, 논리적 흐름을 극대화하여 내용을 가장 효과적으로 전달할 수 있도록 전문적으로 교정하는 것입니다.

    **[구체적인 교정 지침]**
    1.  **문제 정의:** 현 상황의 심각성과 필요성을 강조하고, 문제의 범위와 현행 정책의 한계를 명확히 제시하여 제안의 당위성을 확보하도록 교정합니다.
    2.  **해결 방법:** 방법론을 단계별로 명확히 나누고, '문제 정의'에서 언급된 문제를 어떻게 직접적으로 해결하는지 논리적인 연결 고리를 강화하며, 실현 가능성과 구체성을 높이도록 다듬습니다.
    3.  **기대 효과:** 효과를 정량적/정성적으로 구분하여 제시하고, 수혜자(정책 대상, 사회, 정부)별로 분류하여 임팩트를 강조합니다.
    4.  **전반:** 섹션 간의 논리적 흐름과 유기적인 연결을 매끄럽게 하고, 명확하고 간결하며 공식적인 어휘를 사용합니다.

    **[출력 요구 사항]**
    * 교정된 결과는 **JSON**을 사용하여 **"문제 정의", "해결 방법", "기대 효과"**의 세 가지 섹션으로 출력합니다.

    **[정책 제안서 초안]**

    ## 제목
    {pro_dto.title}

    ## 문제 정의
    {pro_dto.problem}

    ## 해결 방법
    {pro_dto.method}

    ## 기대 효과
    {pro_dto.effect}
    """
    guildlines = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={ "response_mime_type": "application/json",
                 "response_json_schema": HelperResDto.model_json_schema() }
    ).text or "{}"

    return guildlines

@app.get("/examples")
def list_examples(q: Optional[str] = None, page: int = 1):
    PAGE_SIZE = 10
    query = f"%{q}%" if q else "%%"
    offset = (page - 1) * PAGE_SIZE if page > 0 else 0

    with conn.cursor() as cur:
        cur.execute("""SELECT examples.id, title, thumbnail, content, display_name AS region, STRING_AGG(name, ',' ORDER BY name) AS categories, nickname
                    FROM examples, regions, ex_tags, tags, users
                    WHERE examples.rid=regions.id AND examples.id=ex_tags.eid AND ex_tags.tid=tags.id AND examples.uid=users.id AND title LIKE %s
                    GROUP BY (examples.id, display_name, nickname)
                    ORDER BY examples.id DESC
                    OFFSET %s ROWS
                    FETCH NEXT %s ROWS ONLY;""",
                    (query, offset, PAGE_SIZE))
        results = [ { "id": id, "title": title, "thumbnail": thumbnail, "content": content, "region": region, "categories": cat.split(","), "nickname": nickname }
                    for id, title, thumbnail, content, region, cat, nickname in cur.fetchall() ]

    return { "examples": results }

class ExDto(BaseModel):
    rid: int
    title: str
    thumbnail: Optional[str] = None
    content: str
    reference: str
    tags: list[str]

@app.post("/examples")
def create_example(ex_dto: ExDto, authorization: Optional[str] = Header(default=None)):
    if not authorization:
        raise HTTPException(401, "Authorization header not found")
    elif not (token:=parse_token(authorization)):
        raise HTTPException(403, "Authorization failed")

    uid = token["uid"]
    vec = model.encode(ex_dto.content)
    vec_str = f"[{','.join(map(str, vec))}]"
    with conn.cursor() as cur:
        cur.execute("""WITH eids AS (
                    INSERT INTO examples (rid, uid, title, thumbnail, content, reference, vec) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id)
                    INSERT INTO ex_tags (eid, tid) SELECT eids.id, tags.id FROM eids, tags WHERE name IN %s RETURNING eid;""", (ex_dto.rid, uid, ex_dto.title, ex_dto.thumbnail, ex_dto.content, ex_dto.reference, vec_str, tuple(ex_dto.tags)))
        results = cur.fetchall()

    conn.commit()
    if len(results) == 0:
        raise HTTPException(500)
    else:
        return { "eid": results[0][0] }

@app.get("/examples/{eid}")
def get_example(eid: int):
    with conn.cursor() as cur:
        cur.execute("""SELECT examples.id, title, thumbnail, content, display_name AS region, STRING_AGG(name, ',' ORDER BY name) AS categories, nickname
                    FROM examples, regions, ex_tags, tags, users
                    WHERE examples.rid=regions.id AND examples.id=ex_tags.eid AND ex_tags.tid=tags.id AND examples.uid=users.id AND eid=%s
                    GROUP BY (examples.id, display_name, nickname);""", (eid, ))
        results = [ { "id": id, "title": title, "thumbnail": thumbnail, "content": content, "region": region, "categories": cat.split(","), "nickname": nickname }
                    for id, title, thumbnail, content, region, cat, nickname in cur.fetchall() ]

    if len(results) == 0:
        raise HTTPException(404, "No such eid")
    else:
        return results[0]

class AutofillDto(BaseModel):
    url: str

@app.post("/autofill")
def create_autofill(auto_dto: AutofillDto):
    url = auto_dto.url

    try:
        with urlopen(url) as f:
            doc = f.read()
            soup = BeautifulSoup(doc)
            tag = soup.select_one("meta[property='og:image']")
            thumbnail = tag.get("content") if tag is not None else None

        prompt = "첨부된 인터넷 신문 기사를 마크다운(.md) 형식으로 요약해줘. 만일 실패할 경우 아무 문자열도 반환하지 마. "
        result = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Part.from_bytes(data=doc, mime_type="text/html"),
                prompt
            ]
        )

        return { "content": result.text, "thumbnail": thumbnail }
    except BaseException as e:
        raise HTTPException(500, e)
    pass
