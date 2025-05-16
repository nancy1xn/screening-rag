import os
import typing as t
from typing import List

import more_itertools as mit
import MySQLdb
from newsplease.NewsArticle import NewsArticle
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient, models

from screening_rag.custom_types import Crime


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    QDRANT_DOMAIN: str
    MYSQLDB_HOST: str
    MYSQLDB_USER: str
    MYSQLDB_PW: SecretStr
    MYSQLDB_DATABASE: str
    OPENAI_API_KEY: str


settings = Settings()


def reset_and_create_cnn_news_sql_data_storage():
    db = MySQLdb.connect(
        host=settings.MYSQLDB_HOST,
        user=settings.MYSQLDB_USER,
        password=settings.MYSQLDB_PW.get_secret_value(),
    )
    cur = db.cursor()

    cur.execute("CREATE DATABASE my_database")
    db.select_db("my_database")
    cur.execute("""CREATE TABLE CNN_NEWS (
                ID BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, 
                title VARCHAR(300), 
                keyword VARCHAR(300),
                description VARCHAR(3000), 
                maintext MEDIUMTEXT, 
                date_publish DATETIME, 
                url VARCHAR(300), 
                PRIMARY KEY(ID));""")

    cur.execute("""CREATE TABLE CHUNK_CNN_NEWS (
        ID BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        text VARCHAR(1000),
        start_position INT UNSIGNED,
        end_position INT UNSIGNED,
        parent_article_id BIGINT UNSIGNED NOT NULL,
        PRIMARY KEY(ID),
        FOREIGN KEY (parent_article_id) REFERENCES CNN_NEWS(ID));""")
    cur.close()
    db.close()

    qdrant_domain = os.getenv("QDRANT_DOMAIN")
    client = QdrantClient(url=qdrant_domain)
    client.delete_collection(collection_name="cnn_news_chunk_vectors")
    client.create_collection(
        collection_name="cnn_news_chunk_vectors",
        vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
    )


def insert_cnn_news_into_table(keyword: str, news_article: NewsArticle) -> int:
    db = MySQLdb.connect(
        host=settings.MYSQLDB_HOST,
        user=settings.MYSQLDB_USER,
        password=settings.MYSQLDB_PW.get_secret_value(),
        database=settings.MYSQLDB_DATABASE,
    )
    cur = db.cursor()
    cur.execute(
        """INSERT INTO my_database.CNN_NEWS (title, keyword, description, maintext, date_publish, url)
            VALUES (%s, %s, %s, %s, %s, %s)""",
        (
            news_article.title,
            keyword,
            news_article.description,
            news_article.maintext,
            news_article.date_publish,
            news_article.url,
        ),
    )
    article_id = cur.lastrowid
    db.commit()

    cur.execute("select * from my_database.CNN_NEWS where ID =%s", (article_id,))
    for row in cur.fetchall():
        print(row)
    cur.close()
    db.close()

    return article_id


def reset_and_create_crimes_sql_data_storage():
    db = MySQLdb.connect(
        host=settings.MYSQLDB_HOST,
        user=settings.MYSQLDB_USER,
        password=settings.MYSQLDB_PW.get_secret_value(),
        database=settings.MYSQLDB_DATABASE,
    )
    cur = db.cursor()

    cur.execute("""CREATE TABLE CRIME_CNN_NEWS (
                 ID BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, 
                 title VARCHAR(300), 
                 keyword VARCHAR(300),
                 date_publish DATETIME, 
                 time VARCHAR(50), 
                 summary VARCHAR(2000),
                 adverse_info_type VARCHAR(1000), 
                 violated_laws VARCHAR(1000),
                 enforcement_action VARCHAR(1000),              
                 url VARCHAR(1000), 
                 PRIMARY KEY(ID)
                 );
    """)

    cur.execute("""CREATE TABLE SUBJECT_CNN_NEWS (
                ID BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, 
                subject VARCHAR(500),
                parent_crime_id BIGINT UNSIGNED,        
                PRIMARY KEY(ID),
                FOREIGN KEY (parent_crime_id) REFERENCES CRIME_CNN_NEWS(ID)
                );
    """)
    cur.close()
    db.close()


def insert_crime_into_table(keyword: str, news_article: NewsArticle, crime: Crime):
    db = MySQLdb.connect(
        host=settings.MYSQLDB_HOST,
        user=settings.MYSQLDB_USER,
        password=settings.MYSQLDB_PW.get_secret_value(),
        database=settings.MYSQLDB_DATABASE,
    )
    cur = db.cursor()
    crime_adverse_info_type = ",".join(crime.adverse_info_type)
    cur.execute(
        """
        INSERT INTO my_database.CRIME_CNN_NEWS (title, keyword, date_publish, time, summary, adverse_info_type, violated_laws, enforcement_action, url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
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
        ),
    )
    crime.id = cur.lastrowid

    for subject in crime.subjects:
        cur.execute(
            """INSERT INTO my_database.SUBJECT_CNN_NEWS(subject, parent_crime_id) VALUES (%s, %s)""",
            (subject, crime.id),
        )
        db.commit()

    cur.execute("select * from my_database.CRIME_CNN_NEWS where ID =%s", (crime.id,))
    for row in cur.fetchall():
        print(row)
    cur.execute("select * from my_database.SUBJECT_CNN_NEWS")
    for row in cur.fetchall():
        print(row)
    cur.close()
    db.close()


def insert_chunk_table(article_id, chunks):
    db = MySQLdb.connect(
        host=settings.MYSQLDB_HOST,
        user=settings.MYSQLDB_USER,
        password=settings.MYSQLDB_PW.get_secret_value(),
        database=settings.MYSQLDB_DATABASE,
    )
    cur = db.cursor()
    inserted_chunks = []
    for chunk in chunks:
        cur.execute(
            """INSERT INTO my_database.CHUNK_CNN_NEWS (text, parent_article_id)
                VALUES (%s, %s)""",
            (chunk, article_id),
        )
        chunk_id = cur.lastrowid
        inserted_chunks.append((chunk, article_id, chunk_id))

    db.commit()
    cur.execute("select * from my_database.CHUNK_CNN_NEWS")
    for row in cur.fetchall():
        print(row)
    cur.close()
    db.close()
    return inserted_chunks


def get_latest_time_for_cnn_news(keyword: str):
    db = MySQLdb.connect(
        host=settings.MYSQLDB_HOST,
        user=settings.MYSQLDB_USER,
        password=settings.MYSQLDB_PW.get_secret_value(),
        database=settings.MYSQLDB_DATABASE,
    )
    cur = db.cursor()
    cur.execute(
        """SELECT date_publish 
        FROM my_database.CNN_NEWS 
        WHERE keyword = %s 
        ORDER BY date_publish 
        DESC LIMIT 1""",
        (keyword,),
    )
    db.commit()
    latesttime_for_cnn_news = cur.fetchall()
    print(latesttime_for_cnn_news[0][0])
    cur.close()
    db.close()
    return latesttime_for_cnn_news[0][0]


def select_background_grounding_data_from_db(match_ids, final_appendix) -> List[tuple]:
    db = MySQLdb.connect(
        host=settings.MYSQLDB_HOST,
        user=settings.MYSQLDB_USER,
        password=settings.MYSQLDB_PW.get_secret_value(),
        database=settings.MYSQLDB_DATABASE,
    )
    cur = db.cursor()
    for article_id in match_ids:
        query = "select ID, title, url from my_database.CNN_NEWS where ID = %s"
        cur.execute(query, (int(article_id),))
        final_appendix.append(mit.one(cur.fetchall()))
    return final_appendix


def select_distinct_subjects_from_db(subject: str) -> t.List[tuple]:
    db = MySQLdb.connect(
        host=settings.MYSQLDB_HOST,
        user=settings.MYSQLDB_USER,
        password=settings.MYSQLDB_PW.get_secret_value(),
        database=settings.MYSQLDB_DATABASE,
    )
    cur = db.cursor()
    cur.execute("SELECT DISTINCT subject FROM my_database.SUBJECT_CNN_NEWS")
    existing_subjects = cur.fetchall()
    cur.close()
    db.close()
    return existing_subjects


def select_crime_events_grounding_data_from_db(match_ids) -> List[tuple]:
    db = MySQLdb.connect(
        host=settings.MYSQLDB_HOST,
        user=settings.MYSQLDB_USER,
        password=settings.MYSQLDB_PW.get_secret_value(),
        database=settings.MYSQLDB_DATABASE,
    )
    cur = db.cursor()
    final_appendix = []
    for id in match_ids:
        query = "select ID, title, url from my_database.CRIME_CNN_NEWS where ID = %s"
        cur.execute(query, (int(id),))
        for row in cur.fetchall():
            final_appendix.append(row)

    sorted_appendix = sorted(final_appendix, key=lambda x: x[0])
    cur.close()
    db.close()
    return sorted_appendix
