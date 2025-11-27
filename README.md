# pulppuri_api
## setup
requirements: `docker`, `python` >= 3.9, `env.py` file
```sh
# docker image for postgresql and pgvector
PGVECTOR_IMAGE="pgvector/pgvector:pg18-trixie"
docker pull $PGVECTOR_IMAGE
docker run --name pgvector -p 5432:5432 -e POSTGRES_PASSWORD="{find PG_PASSWORD at env.py}" -d $PGVECTOR_IMAGE

# python dependency
pip install -r requirements.txt

# initial data setup
python src/db-setup.py
```

## open
```sh
cd src
uvicorn main:app [--reload] # open server at port 8000
```