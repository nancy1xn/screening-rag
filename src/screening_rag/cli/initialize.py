import typing as t

import requests
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters.sentence_transformers import (
    SentenceTransformersTokenTextSplitter,
)
from newsplease import NewsPlease
from newsplease.NewsArticle import NewsArticle

from screening_rag.custom_types import Crime, NewsSummary, SortingBy
from screening_rag.db import (
    insert_chunk_table,
    insert_cnn_news_into_table,
    insert_crime_into_table,
    reset_and_create_cnn_news_sql_data_storage,
    reset_and_create_crimes_sql_data_storage,
)
from screening_rag.qdrant import (
    process_and_insert_cnn_news_chunks_to_qdrant,
    process_and_insert_crime_to_qdrant,
    reset_and_create_cnn_news_qdrant_data_storage,
    reset_and_create_crime_qdrant_data_storage,
)


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
        yield NewsPlease.from_url(url)


# filter cnn_news and crime_events
def get_crimes_from_summarized_news(
    article: NewsArticle,
) -> t.List[Crime]:
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_model = model.with_structured_output(NewsSummary)
    system = """You are a helpful assistant to check if the contents contains adverse media related to financial crime and help summarize the event, 
            please return boolean: True. If the news is not related to financial crime, please return boolean: False.

            In addition, please list out all financial crimes in the news to summarize the financial crime in terms of the time(ONLY USE "news published date"((newsarticle.date_publish)), the event, 
            the crime type, the direct object of wrongdoing, and the laws or regulations action."""

    news_summary = structured_model.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=str(article.maintext)),
            HumanMessage(content=str(article.date_publish)),
        ]
    )
    news_summary: t.NewsSummary
    if news_summary.is_adverse_news:
        return news_summary.crimes
    else:
        return


def fetch_top_k_cnn_news_crimes(
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
        news_articles = get_cnn_news(keyword, sort_by, page)
        for news in news_articles:
            if count >= amount:
                return news_article_collection
            if crimes := get_crimes_from_summarized_news(news):
                news_article_collection.append((news, crimes))
                count += 1
        page += 1

    return news_article_collection


def chunk_text(maintext: str) -> t.Iterable[str]:
    """Split text into several chunks which are smaller units

    Recursively split the text by characters, and then use a sentence model tokenizer to further split the text into chunk tokens.
    Chunks are yielded one group at a time ,allowing iteration over each chunks group.

    Args:
        text(str): The text to be split into chunks.
    """
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=256, chunk_overlap=50, add_start_index=True
    )
    text_splits = character_splitter.split_text(maintext)
    token_splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=256)

    for section in text_splits:
        yield token_splitter.split_text(section)


def initialize_system(keywords: str, amount: int, sort_by: SortingBy):
    reset_and_create_cnn_news_sql_data_storage()
    reset_and_create_cnn_news_qdrant_data_storage()
    reset_and_create_crimes_sql_data_storage()
    reset_and_create_crime_qdrant_data_storage()

    for keyword in keywords.split(","):
        for news_article, crimes in fetch_top_k_cnn_news_crimes(
            keyword, amount, sort_by
        ):
            article_id = insert_cnn_news_into_table(keyword, news_article)
            chunks = chunk_text(news_article.maintext)
            results = insert_chunk_table(article_id, chunks)
            for chunk, article_id, chunk_id in results:
                process_and_insert_cnn_news_chunks_to_qdrant(
                    chunk, article_id, chunk_id
                )

            for crime in crimes:
                insert_crime_into_table(keyword, news_article, crime)
                process_and_insert_crime_to_qdrant(crime)


def main(keywords, amount, sortby):
    # from argparse import ArgumentParser

    # parser = ArgumentParser()

    # parser.add_argument(
    #     "--keywords",
    #     help="The keywords to search on CNN",
    #     type=str,
    # )
    # parser.add_argument("--amount", help="The amount of the crawled articles", type=int)
    # parser.add_argument(
    #     "-s",
    #     "--sortby",
    #     help="The factor of news ranking",
    #     default=SortingBy.RELEVANCY,
    # )
    # args = parser.parse_args()

    # initialize_system(args.keywords, args.amount, args.sortby)
    initialize_system(keywords, amount, sortby)
