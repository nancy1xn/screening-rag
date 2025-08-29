# import os
from unittest import mock
from unittest.mock import MagicMock
from urllib.parse import urlparse

import psycopg2
import pytest
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

from screening_rag.cli.initialize import (
    fetch_top_k_cnn_news_crimes,
    initialize_system,
    insert_cnn_news_into_table,
    reset_and_create_cnn_news_sql_data_storage,
)
from screening_rag.cli.report_fastapi import app
from screening_rag.custom_types import AdverseInfoType, Crime

client = TestClient(app)


@pytest.fixture(scope="session")
def pg_connection_url():
    with PostgresContainer("pgvector/pgvector:pg17-trixie") as postgres:
        print(postgres.get_connection_url)
        yield postgres.get_connection_url()


def test_initialization(pg_connection_url):
    # insert fake data
    result = urlparse(pg_connection_url)
    username = result.username
    password = result.password
    database = result.path[1:]
    hostname = result.hostname
    test_port = result.port

    # os.environ["AWS_POSTGRESQL_HOST"] = hostname
    # os.environ["AWS_POSTGRESQL_NAME"] = database
    # os.environ["AWS_POSTGRESQL_USER"] = username
    # os.environ["AWS_POSTGRESQL_PW"] = password
    # os.environ["AWS_POSTGRESQL_PORT"] = str(test_port)

    # 關鍵：導入 aws_db 並直接修改已存在的 settings 對象
    from pydantic import SecretStr

    from screening_rag import aws_db

    # 直接修改 settings 對象的屬性
    aws_db.settings.AWS_POSTGRESQL_HOST = hostname
    aws_db.settings.AWS_POSTGRESQL_NAME = database
    aws_db.settings.AWS_POSTGRESQL_USER = username
    aws_db.settings.AWS_POSTGRESQL_PW = SecretStr(password)
    aws_db.settings.AWS_POSTGRESQL_PORT = str(test_port)

    # 準備假的 NewsArticle
    from newsplease import NewsArticle

    a1 = MagicMock(spec=NewsArticle)
    a1.title = "CEO of Binance resigned"
    a1.date_publish = "2025-08-28 00:00:00"
    a1.description = "Changpeng Zhao founded Binance in 2017."
    a1.maintext = "Binance CEO CZ resigned and pled guilty to federal money laundering charges, with a $4.3B US settlement."
    a1.url = "123"

    with mock.patch(
        "screening_rag.cli.initialize.get_cnn_news", return_value=iter([a1])
    ):
        with mock.patch(
            "screening_rag.cli.initialize.get_crimes_from_summarized_news",
            return_value=[
                Crime(
                    time="202508",
                    summary="CEO of Binance, resigned and pleaded guilty to federal money laundering charges.",
                    adverse_info_type=[
                        AdverseInfoType.Money_Laundering_Terrorist_Financing
                    ],
                    subjects=["CEO of Binance", "Binance"],
                    violated_laws="Federal money laundering laws",
                    enforcement_action="CEO of Binance agreed to pay $4.3B US settlement.",
                    id=1,
                )
            ],
        ):
            initialize_system("Binance", 1, "RELEVANCY")
            conn = psycopg2.connect(
                dbname=database,
                user=username,
                password=password,
                host=hostname,
                port=test_port,
            )
            cur = conn.cursor()
            query1 = "SELECT title from CNN_NEWS"
            cur.execute(query1)
            assert cur.fetchall() == [("CEO of Binance resigned",)]

            query2 = "SELECT summary, violated_laws from CRIME_CNN_NEWS"
            cur.execute(query2)
            assert cur.fetchall() == [
                (
                    "CEO of Binance, resigned and pleaded guilty to federal money laundering charges.",
                    "Federal money laundering laws",
                )
            ]

            query3 = "SELECT count(*) from CRIME_CNN_NEWS"
            cur.execute(query3)
            assert cur.fetchall()[0] == (1,)

            cur.close()
            conn.close()

    # response = client.get("/search?entity_name=Binance")
    # assert response.status_code == 200
    # response = client.get("/search?entity_name=ABC Company")
    # assert response.status_code == 200


def test_fetch_top_k():
    # 準備假的 NewsArticle

    with mock.patch(
        "screening_rag.cli.initialize.get_cnn_news", return_value=iter(["Hello"])
    ):
        with mock.patch(
            "screening_rag.cli.initialize.get_crimes_from_summarized_news",
            return_value=[
                Crime(
                    time="202598",
                    summary="CEO of Binance, resigned and pleaded guilty to federal money laundering charges.",
                    adverse_info_type=[
                        AdverseInfoType.Money_Laundering_Terrorist_Financing
                    ],
                    subjects=["CEO of Binance", "Binance"],
                    violated_laws="Federal money laundering laws",
                    enforcement_action="CEO of Binance agreed to pay $4.3B US settlement.",
                    id=1,
                )
            ],
        ):
            news_article_collection = fetch_top_k_cnn_news_crimes(
                "Binance", 1, "RELEVANCY"
            )
            assert news_article_collection == [
                (
                    "Hello",
                    [
                        Crime(
                            time="202598",
                            summary="CEO of Binance, resigned and pleaded guilty to federal money laundering charges.",
                            adverse_info_type=[
                                AdverseInfoType.Money_Laundering_Terrorist_Financing
                            ],
                            subjects=["CEO of Binance", "Binance"],
                            violated_laws="Federal money laundering laws",
                            enforcement_action="CEO of Binance agreed to pay $4.3B US settlement.",
                            id=1,
                        )
                    ],
                )
            ]


def test_insert_cnn_news_into_table(pg_connection_url):
    result = urlparse(pg_connection_url)
    username = result.username
    password = result.password
    database = result.path[1:]
    hostname = result.hostname
    test_port = result.port
    from pydantic import SecretStr

    from screening_rag import aws_db

    # 直接修改 settings 對象的屬性
    aws_db.settings.AWS_POSTGRESQL_HOST = hostname
    aws_db.settings.AWS_POSTGRESQL_NAME = database
    aws_db.settings.AWS_POSTGRESQL_USER = username
    aws_db.settings.AWS_POSTGRESQL_PW = SecretStr(password)
    aws_db.settings.AWS_POSTGRESQL_PORT = str(test_port)

    class News:
        def __init__(self, title, description, maintext, date_publish, url):
            self.title = title
            self.description = description
            self.maintext = maintext
            self.date_publish = date_publish
            self.url = url

    reset_and_create_cnn_news_sql_data_storage()
    insert_cnn_news_into_table(
        "Binance",
        News(
            title="CEO of Binance resigned",
            date_publish="2025-08-28 00:00:00",
            description="Changpeng Zhao founded Binance in 2017.",
            maintext="Binance CEO CZ resigned and pled guilty to federal money laundering charges, with a $4.3B US settlement.",
            url="123",
        ),
    )
    conn = psycopg2.connect(
        dbname=database,
        user=username,
        password=password,
        host=hostname,
        port=test_port,
    )
    cur = conn.cursor()
    query = "SELECT title, url, keyword FROM CNN_NEWS;"
    cur.execute(query)
    assert cur.fetchall() == [
        (
            "CEO of Binance resigned",
            "123",
            "Binance",
        )
    ]
    cur.close()
    conn.close()
