import typing as t
from typing import List

import more_itertools as mit
import psycopg2
from langchain_openai import OpenAIEmbeddings
from newsplease.NewsArticle import NewsArticle
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client.http.models.models import QueryResponse

from screening_rag.custom_types import Crime, SimilarSubjects


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    AWS_POSTGRESQL_HOST: str
    AWS_POSTGRESQL_NAME: str
    AWS_POSTGRESQL_USER: str
    AWS_POSTGRESQL_PW: SecretStr
    AWS_POSTGRESQL_PORT: str
    OPENAI_API_KEY: str


settings = Settings()


def reset_and_create_cnn_news_sql_data_storage():
    conn = psycopg2.connect(
        dbname=settings.AWS_POSTGRESQL_NAME,
        user=settings.AWS_POSTGRESQL_USER,
        password=settings.AWS_POSTGRESQL_PW.get_secret_value(),
        host=settings.AWS_POSTGRESQL_HOST,
        port=settings.AWS_POSTGRESQL_PORT,
    )

    cur = conn.cursor()
    conn.autocommit = True

    cur.execute("DROP TABLE IF EXISTS CHUNK_CNN_NEWS")
    cur.execute("DROP TABLE IF EXISTS CNN_NEWS")

    # enable extension vector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cur.execute("""CREATE TABLE CNN_NEWS (
                ID BIGSERIAL not null PRIMARY KEY CHECK (id > 0), 
                title VARCHAR(300),
                keyword VARCHAR(300),
                description VARCHAR(3000),
                maintext TEXT,
                date_publish timestamp,
                url VARCHAR(300)
                );""")

    cur.execute("""CREATE TABLE CHUNK_CNN_NEWS (
                ID BIGSERIAL not null PRIMARY KEY CHECK (id > 0), 
                text VARCHAR(1000),
                parent_article_id BIGINT not null REFERENCES CNN_NEWS(ID),
                embedding vector(3072)
                );""")
    conn.commit()
    cur.close()
    conn.close()


def insert_cnn_news_into_table(keyword: str, news_article: NewsArticle) -> int:
    conn = psycopg2.connect(
        dbname=settings.AWS_POSTGRESQL_NAME,
        user=settings.AWS_POSTGRESQL_USER,
        password=settings.AWS_POSTGRESQL_PW.get_secret_value(),
        host=settings.AWS_POSTGRESQL_HOST,
        port=settings.AWS_POSTGRESQL_PORT,
    )

    cur = conn.cursor()
    conn.autocommit = True

    cur.execute(
        "INSERT INTO CNN_NEWS (title, keyword, description, maintext, date_publish, url) VALUES (%s, %s, %s, %s, %s, %s) RETURNING ID;",
        (
            news_article.title,
            keyword,
            news_article.description,
            news_article.maintext,
            news_article.date_publish,
            news_article.url,
        ),
    )
    article_id = cur.fetchone()[0]
    cur.execute("select * from CNN_NEWS where ID =%s", (article_id,))
    for row in cur.fetchall():
        print(row)

    conn.commit()
    cur.close()
    conn.close()

    return article_id


def reset_and_create_crimes_sql_data_storage():
    conn = psycopg2.connect(
        dbname=settings.AWS_POSTGRESQL_NAME,
        user=settings.AWS_POSTGRESQL_USER,
        password=settings.AWS_POSTGRESQL_PW.get_secret_value(),
        host=settings.AWS_POSTGRESQL_HOST,
        port=settings.AWS_POSTGRESQL_PORT,
    )

    cur = conn.cursor()
    conn.autocommit = True

    cur.execute("DROP TABLE IF EXISTS SUBJECT_CNN_NEWS")
    cur.execute("DROP TABLE IF EXISTS CRIME_CNN_NEWS")

    # enable extension vector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cur.execute("""CREATE TABLE CRIME_CNN_NEWS (
                ID BIGSERIAL not null PRIMARY KEY CHECK (id > 0), 
                title VARCHAR(300),
                keyword VARCHAR(300),
                date_publish timestamp,
                time VARCHAR(50),
                summary VARCHAR(2000),
                adverse_info_type VARCHAR(1000), 
                violated_laws VARCHAR(1000),
                enforcement_action VARCHAR(1000),              
                url VARCHAR(1000),
                embedding vector(3072)
                );""")

    cur.execute("""CREATE TABLE SUBJECT_CNN_NEWS (
                ID BIGSERIAL not null PRIMARY KEY CHECK (id > 0), 
                subject VARCHAR(500),
                parent_crime_id BIGINT not null REFERENCES CRIME_CNN_NEWS(ID)
                );""")
    conn.commit()
    cur.close()
    conn.close()


def insert_crime_into_table(keyword: str, news_article: NewsArticle, crime: Crime):
    conn = psycopg2.connect(
        dbname=settings.AWS_POSTGRESQL_NAME,
        user=settings.AWS_POSTGRESQL_USER,
        password=settings.AWS_POSTGRESQL_PW.get_secret_value(),
        host=settings.AWS_POSTGRESQL_HOST,
        port=settings.AWS_POSTGRESQL_PORT,
    )

    cur = conn.cursor()
    conn.autocommit = True

    crime_adverse_info_type = ",".join(crime.adverse_info_type)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
    crime_openai_vectors = embeddings.embed_documents([str(crime.summary)])
    crime_openai_vectors: t.List[t.List[float]]
    crime_openai_vector = crime_openai_vectors[0]

    cur.execute(
        "INSERT INTO CRIME_CNN_NEWS (title, keyword, date_publish, time, summary, adverse_info_type, violated_laws, enforcement_action, url, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING ID;",
        (
            news_article.title,
            keyword,
            news_article.date_publish,
            crime.time,
            crime.summary,
            crime_adverse_info_type,
            crime.violated_laws,
            crime.enforcement_action,
            news_article.url,
            crime_openai_vector,
        ),
    )
    crime.id = cur.fetchone()[0]

    for subject in crime.subjects:
        cur.execute(
            """INSERT INTO SUBJECT_CNN_NEWS(subject, parent_crime_id) VALUES (%s, %s)""",
            (subject, crime.id),
        )
        conn.commit()

    cur.execute("select * from CRIME_CNN_NEWS where ID =%s", (crime.id,))
    for row in cur.fetchall():
        print(row)
    cur.execute("select * from SUBJECT_CNN_NEWS")
    for row in cur.fetchall():
        print(row)
    cur.close()
    conn.close()


def insert_chunk_table(article_id, chunks):
    conn = psycopg2.connect(
        dbname=settings.AWS_POSTGRESQL_NAME,
        user=settings.AWS_POSTGRESQL_USER,
        password=settings.AWS_POSTGRESQL_PW.get_secret_value(),
        host=settings.AWS_POSTGRESQL_HOST,
        port=settings.AWS_POSTGRESQL_PORT,
    )

    cur = conn.cursor()
    conn.autocommit = True
    inserted_chunks = []
    for chunk in chunks:
        print(chunk)
        print(type(chunk))
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
        text_openai_vectors = embeddings.embed_documents([chunk[0]])
        text_openai_vectors: t.List[List[float]]
        each_text_openai_vectors = text_openai_vectors[0]

        cur.execute(
            """INSERT INTO CHUNK_CNN_NEWS (text, parent_article_id, embedding)
                VALUES (%s, %s, %s) RETURNING ID;""",
            (chunk, article_id, each_text_openai_vectors),
        )
        chunk_id = cur.fetchone()[0]
        inserted_chunks.append((chunk, article_id, chunk_id))

    conn.commit()
    cur.execute("select * from CHUNK_CNN_NEWS")
    for row in cur.fetchall():
        print(row)
    cur.close()
    conn.close()
    return inserted_chunks


def get_latest_time_for_cnn_news(keyword: str):
    conn = psycopg2.connect(
        dbname=settings.AWS_POSTGRESQL_NAME,
        user=settings.AWS_POSTGRESQL_USER,
        password=settings.AWS_POSTGRESQL_PW.get_secret_value(),
        host=settings.AWS_POSTGRESQL_HOST,
        port=settings.AWS_POSTGRESQL_PORT,
    )

    cur = conn.cursor()
    cur.execute(
        """SELECT date_publish 
        FROM CNN_NEWS 
        WHERE keyword = %s 
        ORDER BY date_publish DESC 
        LIMIT 1""",
        (keyword,),
    )
    conn.commit()
    latesttime_for_cnn_news = cur.fetchall()
    print(latesttime_for_cnn_news[0][0])
    cur.close()
    conn.close()
    return latesttime_for_cnn_news[0][0]


def select_background_grounding_data_from_db(match_ids, final_appendix) -> List[tuple]:
    conn = psycopg2.connect(
        dbname=settings.AWS_POSTGRESQL_NAME,
        user=settings.AWS_POSTGRESQL_USER,
        password=settings.AWS_POSTGRESQL_PW.get_secret_value(),
        host=settings.AWS_POSTGRESQL_HOST,
        port=settings.AWS_POSTGRESQL_PORT,
    )

    cur = conn.cursor()
    for article_id in match_ids:
        query = "select ID, title, url from CNN_NEWS where ID = %s"
        cur.execute(query, (int(article_id),))
        final_appendix.append(mit.one(cur.fetchall()))

    cur.close()
    conn.close()

    return final_appendix


def select_distinct_subjects_from_db(subject: str) -> t.List[tuple]:
    conn = psycopg2.connect(
        dbname=settings.AWS_POSTGRESQL_NAME,
        user=settings.AWS_POSTGRESQL_USER,
        password=settings.AWS_POSTGRESQL_PW.get_secret_value(),
        host=settings.AWS_POSTGRESQL_HOST,
        port=settings.AWS_POSTGRESQL_PORT,
    )

    cur = conn.cursor()
    cur.execute("SELECT DISTINCT subject FROM SUBJECT_CNN_NEWS")
    existing_subjects = cur.fetchall()
    cur.close()
    conn.close()
    return existing_subjects


def select_crime_events_grounding_data_from_db(match_ids) -> List[tuple]:
    conn = psycopg2.connect(
        dbname=settings.AWS_POSTGRESQL_NAME,
        user=settings.AWS_POSTGRESQL_USER,
        password=settings.AWS_POSTGRESQL_PW.get_secret_value(),
        host=settings.AWS_POSTGRESQL_HOST,
        port=settings.AWS_POSTGRESQL_PORT,
    )

    cur = conn.cursor()
    final_appendix = []
    for id in match_ids:
        query = "select ID, title, url from CRIME_CNN_NEWS where ID = %s"
        cur.execute(query, (int(id),))
        for row in cur.fetchall():
            final_appendix.append(row)

    sorted_appendix = sorted(final_appendix, key=lambda x: x[0])
    cur.close()
    conn.close()
    return sorted_appendix


def get_crime_points_similar_to_embedding(
    query: str,
    limit: int,
    embedding_model: t.Optional[str] = "text-embedding-3-large",
    dimentions: t.Optional[int] = 3072,
    score_threshold: t.Optional[float] = 0.5,
    extra_conditions: List[SimilarSubjects] = None,
) -> QueryResponse:
    conn = psycopg2.connect(
        dbname=settings.AWS_POSTGRESQL_NAME,
        user=settings.AWS_POSTGRESQL_USER,
        password=settings.AWS_POSTGRESQL_PW.get_secret_value(),
        host=settings.AWS_POSTGRESQL_HOST,
        port=settings.AWS_POSTGRESQL_PORT,
    )

    cur = conn.cursor()
    conn.autocommit = True

    embeddings = OpenAIEmbeddings(model=embedding_model, dimensions=dimentions)
    question_vectors = embeddings.embed_documents([query])
    question_vectors: t.List[List[float]]
    question_vector = question_vectors[0]

    query = """
    SELECT DISTINCT
        c.id, c.time, c.summary, c.adverse_info_type, c.violated_laws, c.enforcement_action, array_agg(DISTINCT s.subject) AS subjects, 1 - (c.embedding <=> %s::vector) as similarity_score
    FROM CRIME_CNN_NEWS AS c
    JOIN SUBJECT_CNN_NEWS s ON c.id = s.parent_crime_id
    WHERE 1 - (c.embedding <=> %s::vector) > %s
    AND s.subject = ANY(%s)
    GROUP BY c.id, c.time, c.summary, c.adverse_info_type, c.violated_laws, c.enforcement_action, similarity_score
    ORDER BY similarity_score DESC
    LIMIT %s;
        """

    cur.execute(
        query,
        (
            question_vector,
            question_vector,
            score_threshold,
            extra_conditions,
            limit,
        ),
    )

    return cur.fetchall()


def get_chunks_points_similar_to_embedding(
    query: str,
    limit: int,
    embedding_model: t.Optional[str] = "text-embedding-3-large",
    dimentions: t.Optional[int] = 3072,
    score_threshold: t.Optional[float] = 0,
):
    conn = psycopg2.connect(
        dbname=settings.AWS_POSTGRESQL_NAME,
        user=settings.AWS_POSTGRESQL_USER,
        password=settings.AWS_POSTGRESQL_PW.get_secret_value(),
        host=settings.AWS_POSTGRESQL_HOST,
        port=settings.AWS_POSTGRESQL_PORT,
    )

    cur = conn.cursor()
    conn.autocommit = True

    embeddings = OpenAIEmbeddings(model=embedding_model, dimensions=dimentions)
    question_vector = mit.one(embeddings.embed_documents([query]))
    question_vector: t.List[List[float]]

    query = """
            SELECT
            c.ID, c.parent_article_id, c.text
            FROM CHUNK_CNN_NEWS AS c
            WHERE 1 - (c.embedding <=> %s::vector) > %s
            ORDER BY 1 - (c.embedding <=> %s::vector) DESC
            LIMIT %s;
            """

    cur.execute(
        query,
        (
            question_vector,
            score_threshold,
            question_vector,
            limit,
        ),
    )

    return cur.fetchall()
