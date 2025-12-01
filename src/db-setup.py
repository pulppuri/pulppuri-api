import csv

import psycopg2
from psycopg2.extensions import connection

import env


def setup_regions(conn: connection):
    with open("data/regions2.csv") as f:
        _ = f.readline()
        cur = conn.cursor()
        reader = csv.reader(f)

        for lv0, lv1, lv2, aux in reader:
            cur.execute("INSERT INTO regions (lv0, full_name, display_name) VALUES (%s, %s, %s) ON CONFLICT (full_name) DO NOTHING;", (lv0, lv0, lv0))
            full_name = " ".join(x for x in (lv0, lv1) if x)
            cur.execute("INSERT INTO regions (lv0, lv1, full_name, display_name) VALUES (%s, %s, %s, %s) ON CONFLICT (full_name) DO NOTHING;", (lv0, lv1, full_name, lv1))
            full_name = " ".join(x for x in (lv0, lv1, lv2) if x)
            cur.execute("INSERT INTO regions (lv0, lv1, lv2, aux, full_name, display_name) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (full_name) DO NOTHING;", (lv0, lv1, lv2, aux, full_name, lv2))

        conn.commit()


def setup_examples(conn: connection):
    cur = conn.cursor()
    with open("data/db3.csv") as f_db, open("data/vecs.csv") as f_vecs:
        _ = f_db.readline()
        s = set()
        reader = zip(csv.reader(f_db), csv.reader(f_vecs))
        reader = filter(lambda x: x[0][0], reader)
        for (_, region, tags, title, bg, ct, ef, ref), vec in reader:
            if region.endswith("(광역)"):
                region = region[:-4-1]
            elif region.endswith("특례시"):
                region = region[:-3] + "시"

            content = "\n".join([
                "### 배경 및 필요성",
                bg,
                "",
                "### 추진 내용",
                ct,
                "",
                "### 정책 효과",
                ef
            ])
            vec = f"""[{",".join(vec)}]"""
            tags = tuple(tags.split())
            cur.execute("""WITH eids AS (
                        INSERT INTO examples (rid, uid, title, content, reference, vec) SELECT (SELECT id FROM regions WHERE full_name=%s), %s, %s, %s, %s, %s RETURNING id)
                        INSERT INTO ex_tags (eid, tid) SELECT eids.id, tags.id FROM eids, tags WHERE name IN %s;""", (region, 1, title, content, ref, vec, tags))


if __name__ == "__main__":
    conn = psycopg2.connect(host=env.PG_HOST,
                            port=env.PG_PORT,
                            database=env.PG_DATABASE,
                            user=env.PG_USERNAME,
                            password=env.PG_PASSWORD)
    cur = conn.cursor()

    # pgvector setup
    cur.execute("CREATE EXTENSION vector;")

    # table setup
    cur.execute("""CREATE TABLE regions (
                id SERIAL PRIMARY KEY,
                lv0 TEXT NOT NULL,
                lv1 TEXT,
                lv2 TEXT,
                aux TEXT,
                full_name TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL
    );""")
    cur.execute("""CREATE TABLE tags (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL
    );""")
    cur.execute("""CREATE TABLE users (
                id SERIAL PRIMARY KEY,
                age INTEGER,
                job TEXT,
                rid INTEGER,
                gender TEXT,
                nickname TEXT
    );""")
    cur.execute("""CREATE TABLE examples (
                id SERIAL PRIMARY KEY,
                rid INTEGER NOT NULL,
                uid INTEGER NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                reference TEXT NOT NULL,
                vec VECTOR(1024) NOT NULL,
                read_cnt INTEGER NOT NULL DEFAULT 0
    );""")
    cur.execute("""CREATE TABLE ex_tags (
                eid INTEGER,
                tid INTEGER,

                PRIMARY KEY (eid, tid)
    );""")
    cur.execute("""CREATE TABLE proposals (
                id SERIAL PRIMARY KEY,
                eid INTEGER,
                rid INTEGER NOT NULL,
                uid INTEGER NOT NULL,
                title TEXT NOT NULL,
                problem TEXT NOT NULL,
                method TEXT NOT NULL,
                effect TEXT NOT NULL,
                read_cnt INTEGER NOT NULL DEFAULT 0
    );""")
    cur.execute("""CREATE TABLE pr_tags (
                pid INTEGER,
                tid INTEGER,

                PRIMARY KEY (pid, tid)
    );""")

    # index setup
    # cur.execute("CREATE INDEX ex_vecs_cos ON ex_vecs USING hnsw (vec vector_cosine_ops);")

    conn.commit()

    # data setup
    cur.execute("""INSERT INTO tags (name) VALUES
                ('교육'), ('교통'), ('농촌'), ('문화'), ('복지'), ('주거'), ('청년');""")
    setup_regions(conn)
    cur.execute("""INSERT INTO users (age, job, rid, gender, nickname) VALUES (20, '기자', 11859, '남성', '옥천신문');""")
    setup_examples(conn)
    conn.commit()
