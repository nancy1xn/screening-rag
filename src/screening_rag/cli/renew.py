import typing as t
from datetime import datetime

from newsplease.NewsArticle import NewsArticle

from screening_rag.cli.initialize import (
    chunk_text,
    get_cnn_news,
    get_crimes_from_summarized_news,
)
from screening_rag.custom_types import Crime, SortingBy
from screening_rag.db import (
    get_latest_time_for_cnn_news,
    insert_chunk_table,
    insert_cnn_news_into_table,
    insert_crime_into_table,
)
from screening_rag.qdrant import (
    process_and_insert_cnn_news_chunks_to_qdrant,
    process_and_insert_crime_to_qdrant,
)


def fetch_latest_cnn_news_crimes(
    keyword: str,
    sorting_by,
    latesttime: datetime,
) -> t.List[t.Tuple[NewsArticle, t.List[Crime]]]:
    page = 1
    news_article_collection = []

    while True:
        news_articles = get_cnn_news(keyword, sorting_by, page)
        for news in news_articles:
            if news.date_publish <= latesttime:
                return news_article_collection
            if crimes := get_crimes_from_summarized_news(news):
                news_article_collection.append((news, crimes))

        page += 1


def renew_system(keywords: str, sort_by: SortingBy):
    for keyword in keywords.split(","):
        latesttime_for_cnn_news = get_latest_time_for_cnn_news(keyword)
        latesttime_for_cnn_news: t.Tuple[t.Tuple[datetime]]

        for news_article, crimes in fetch_latest_cnn_news_crimes(
            keyword, sort_by, datetime(2025, 4, 25, 00, 00, 0)
        ):
            # for news_article, crimes in fetch_latest_cnn_news_crimes(
            #     keyword, sort_by, latesttime_for_cnn_news
            # ):
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


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "--keywords",
        help="The keywords to search on CNN",
        type=str,
    )
    parser.add_argument("--amount", help="The amount of the crawled articles", type=int)
    parser.add_argument(
        "-s",
        "--sortby",
        help="The factor of news ranking",
        default=SortingBy.NEWEST,
    )
    args = parser.parse_args()
    # for keyword in args.keywords.split(","):
    renew_system(args.keywords, args.sortby)
