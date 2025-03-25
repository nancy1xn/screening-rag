import typing as t
from datetime import datetime
from enum import Enum

import MySQLdb
import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from newsplease import NewsPlease
from newsplease.NewsArticle import NewsArticle
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from screening_rag.preprocess import cnn_crime_event
from screening_rag.preprocess.cnn_crime_event import Crime
from screening_rag.preprocess.cnn_news_chunking import insert_chunk_table


class NewsSummary(BaseModel):
    """
    1.Check if the media content is considered as adverse media.
    2.List out and provide summary for all financial crimes in the news.

    -Define the guidelines for checking if the media contents is an adverse media related to financial crime.
     The result will be 'True' if adverse media is found, and 'False' otherwise.
    -List out all financial crimes in the news to summarize the financial crime in terms of the time, the event, the crime type, the direct object of wrongdoing, and the laws or regulations action.

    Attributes:
        is_adverse_news: A boolean indicating if the media content is an adverse media or not.
        crimes: list of 'Class Crime' indicating all financial crimes summary reported in the news.")

    """

    is_adverse_news: bool = Field(
        description="""Please determine if the news mentions the search object (input keyword) that is related to financial crime.
                    Please only return as format boolean (True/False) and determine whether the news related to financial crime accused of committing by the search object (input keyword) as per the below criteria:

                    (1) If the news is related to financial crime, please return boolean: True
                    (2) If the news is not related to financial crime, please return boolean: False

                    **Criteria for Financial Crime**:
                    Financial crime includes the following categories (Media coverage triggers: Regulatory Enforcement Actions, Convictions, Ongoing investigations, or allegations related to these):

                    1. Money laundering
                    2. Bribery and corruption
                    3. Fraud or weakness in fraud prevention controls
                    4. Stock exchange irregularities, insider trading, market manipulation
                    5. Accounting irregularities
                    6. Tax evasion and other tax crimes (related to direct and indirect taxes)
                    7. Regulatory enforcement actions against entities in the regulated sector with links to the client
                    8. Major sanctions breaches (including dealings with or othr involvement with Sanction countries or Territories or Countries Subject to Extensive Sanction Regimes)
                    9. Terrorism (including terrorist financing)
                    10. Illicit trafficking in narcotic drugs and psychotropic substances, arms trafficking or stolen goods
                    11. Smuggling(including in relation to customs and excise duties and taxes)
                    12. Human trafficking or migrant smuggling
                    13. Sexual exploitation of child labor
                    14. Extortion, counterfeiting, forgery, piracy of products
                    15. Organized crime or racketeering
                    16. Benefiting from serious offences (e.g., kidnapping, illegal restraint, hostage-taking, robbery, theft, murder, or causing frievous bodily injury)
                    17. Benefiting from environmental crimes
                    18. Benefiting from other unethical or criminal behavior

                    **Search Object**:
                    The search object refers to the keyword used to search for relevant news. In this case, it would be the term provided via a search request, for example:
                    `requests.get('https://search.prod.di.api.cnn.io/content', params={{'q': keyword}})`
                    """
    )

    crimes: t.List[Crime] = Field(
        description="Please list all financial crime events reported in the news and summarize them in response to the questions defined in the 'Class Crime' section."
    )


class SortingBy(str, Enum):
    """Represent different sorting logics for media.

    Use the enum to categorize sorting logics for media by "newest" or "relevance".

    Attributes:
        NEWEST: The search criteria to sort media by the latest to the oldest.
        RELEVANCY: The search criteria to sort media by the most relevant to the least relevant.
    """

    NEWEST = "newest"
    RELEVANCY = "relevance"


# get url from cnn website
def get_cnn_news(keyword: str, sort_by: SortingBy, page) -> t.Iterable[NewsArticle]:
    size_per_page = 3

    web = requests.get(
        "https://search.prod.di.api.cnn.io/content",
        params={
            "q": keyword,
            "size": size_per_page,
            "sort": sort_by,
            "from": (page - 1) * 3,
            "page": page,
            "request_id": "stellar-search-19c44161-fd1e-4aff-8957-6316363aaa0e",
            "site": "cnn",
        },
    )
    news_collection = web.json().get("result")
    for i, news in enumerate(news_collection):
        if news["type"] == "VideoObject":
            continue
        url = news["path"]
        article = NewsPlease.from_url(url)
        yield article


# filter cnn_news and crime_events
def handle_news_and_crimes(
    article: NewsArticle,
) -> t.Iterable[NewsArticle]:
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_model = model.with_structured_output(NewsSummary)
    system = """You are a helpful assistant to check if the contents contains adverse media related to financial crime and help summarize the event, 
            please return boolean: True. If the news is not related to financial crime, please return boolean: False.

            In addition, please list out all financial crimes in the news to summarize the financial crime in terms of the time(ONLY USE "news published date"((newsarticle.date_publish)), the event, 
            the crime type, the direct object of wrongdoing, and the laws or regulations action."""

    news_summary = structured_model.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=article.maintext),
            HumanMessage(content=str(article.date_publish)),
        ]
    )
    news_summary: t.NewsSummary
    if news_summary.is_adverse_news:
        return news_summary.crimes
    else:
        return


def cnn_news_and_crimes_pipeline(
    keyword: str,
    amount: int,
    sort_by: SortingBy,
) -> t.List[t.Tuple[NewsArticle, t.List[Crime]]]:
    """Retrieve NewsArticle Objects related to financial crime.

    Yield each NewsArticle Object based on CNN's official website and that are related to financial crime one at a time, allowing iteration over each NewsArticle Object.

    Args:
        keyword(str): The keyword used to search CNN news.
        amount(int): The number of articles to be yielded.
        sort_by(SortingBy): The search critera to sort media by "newest" or "relevance". If SortingBy.NEWEST: Sort media by the latest to the oldest.
                            If SortingBy.RELEVANCY: Sort media by the most relevant to the least relevant.
    """
    count = 0
    page = 1
    news_article_collection = []

    while count < amount:
        news_article = get_cnn_news(keyword, sort_by, page)
        for news in news_article:
            if count >= amount:
                return news_article_collection
            if crimes := handle_news_and_crimes(news):
                news_article_collection.append((news, crimes))
                count = len(news_article_collection)
        page += 1

    return news_article_collection


def reset_and_create_cnn_news_data_storage():
    db = MySQLdb.connect(
        host="127.0.0.1", user="root", password="my-secret-pw", database="my_database"
    )
    cur = db.cursor()

    cur.execute("DROP TABLE CHUNK_CNN_NEWS;")
    cur.execute("DROP TABLE CNN_NEWS;")
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

    client = QdrantClient(url="http://localhost:6333")
    client.create_collection(
        collection_name="cnn_news_chunk_vectors",
        vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
    )


def insert_cnn_news_into_table(keyword: str, news_article: NewsArticle):
    db = MySQLdb.connect(
        host="127.0.0.1", user="root", password="my-secret-pw", database="my_database"
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

    return news_article, article_id


def reset_and_create_crimes_data_storage():
    db = MySQLdb.connect(
        host="127.0.0.1", user="root", password="my-secret-pw", database="my_database"
    )
    cur = db.cursor()

    cur.execute("DROP TABLE SUBJECT_CNN_NEWS;")
    cur.execute("DROP TABLE CRIME_CNN_NEWS ;")

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

    client = QdrantClient(url="http://localhost:6333")
    client.create_collection(
        collection_name="crime_cnn_news_vectors",
        vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
    )


def insert_crime_into_table(keyword: str, news_article: NewsArticle, crime: Crime):
    db = MySQLdb.connect(
        host="127.0.0.1", user="root", password="my-secret-pw", database="my_database"
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


def get_latest_time_for_cnn_news(keyword: t.List[str]):
    db = MySQLdb.connect(
        host="127.0.0.1", user="root", password="my-secret-pw", database="my_database"
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

    cur.close()
    db.close()
    return latesttime_for_cnn_news


def handle_and_renew_news_and_crimes(
    article: NewsArticle,
) -> t.List[Crime]:
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_model = model.with_structured_output(NewsSummary)

    system = """You are a helpful assistant to check if the contents contains adverse media related to financial crime and help summarize the event, 
                please return boolean: True. If the news is not related to financial crime, please return boolean: False.

                In addition, please list out all financial crimes in the news to summarize the financial crime in terms of the time (ONLY USE "news published date"((newsarticle.date_publish))）, the event, 
                the crime type, the direct object of wrongdoing, and the laws or regulations action."""

    news_summary = structured_model.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=article.maintext),
            HumanMessage(content=str(article.date_publish)),
        ]
    )
    news_summary: NewsSummary
    if news_summary.is_adverse_news:
        return news_summary.crimes


def renew_cnn_news_and_crimes_pipeline(
    keyword: str, sorting_by, latesttime: datetime
) -> t.List[t.Tuple[NewsArticle, t.List[Crime]]]:
    page = 1
    news_article_collection = []

    while True:
        news_article = get_cnn_news(keyword, sorting_by, page)
        for news in news_article:
            if news.date_publish <= latesttime:
                return news_article_collection
            if crimes := handle_and_renew_news_and_crimes(news):
                news_article_collection.append((news, crimes))

        page += 1


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["mode_initialize", "mode_renew"],
        required=True,
        help="choose mode_initialize or mode_renew",
    )
    parser.add_argument("--keyword", help="The keyword to search on CNN", type=str)
    parser.add_argument("--amount", help="The amount of the crawled articles", type=int)
    parser.add_argument(
        "-s",
        "--sort-by",
        help="The factor of news ranking",
        default=SortingBy.RELEVANCY,
    )
    args = parser.parse_args()

    if args.mode == "mode_initialize":
        keywords = ["JP Morgan financial crime"]
        for keyword in keywords:
            downloaded_news_and_crimes = cnn_news_and_crimes_pipeline(
                keyword, args.amount, args.sort_by
            )

            reset_and_create_cnn_news_data_storage()
            reset_and_create_crimes_data_storage()

            for news_and_crimes in downloaded_news_and_crimes:
                news_article, crimes = news_and_crimes
                news_article, article_id = insert_cnn_news_into_table(
                    keyword, news_article
                )
                insert_chunk_table(news_article, article_id)

                for crime in crimes:
                    insert_crime_into_table(args.keyword, news_article, crime)
                    cnn_crime_event.insert_to_qdrant(crime)

    if args.mode == "mode_renew":
        keywords = ["JP Morgan financial crime"]
        for keyword in keywords:
            latesttime_for_cnn_news = get_latest_time_for_cnn_news(keyword)
            latesttime_for_cnn_news: t.Tuple[t.Tuple[datetime]]
            # renewed_news_and_crimes = renew_cnn_news_and_crimes_pipeline(keyword, SortingBy.NEWEST, datetime(2025, 3, 12, 00, 00, 0))
            renewed_news_and_crimes = renew_cnn_news_and_crimes_pipeline(
                keyword, SortingBy.NEWEST, latesttime_for_cnn_news[0][0]
            )
            for news_and_crimes in renewed_news_and_crimes:
                news_article, crimes = news_and_crimes
                news_article, article_id = insert_cnn_news_into_table(
                    keyword, news_article
                )
                insert_chunk_table(news_article, article_id)

                for crime in crimes:
                    insert_crime_into_table(args.keyword, news_article, crime)
                    cnn_crime_event.insert_to_qdrant(crime)


# news collection
# CURL -L -X GET 'http://localhost:6333/collections/cnn_news_chunk_vectors/points/1'
# CURL -X DELETE "http://localhost:6333/collections/cnn_news_chunk_vectors"

# crime collection
# CURL -L -X GET 'http://localhost:6333/collections/crime_cnn_news_vectors/points/1'
# CURL -X DELETE "http://localhost:6333/collections/crime_cnn_news_vectors"
