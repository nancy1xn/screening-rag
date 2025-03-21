from enum import Enum
import requests
import typing as t
from newsplease.NewsArticle import NewsArticle
from newsplease import NewsPlease
import MySQLdb
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from screening_rag.preprocess.cnn_news_chunking import insert_chunk_table
from qdrant_client import QdrantClient, models

class IsAdverseMedia(BaseModel):
    """Check if the media content is considered as adverse media.

    Define the guidelines for checking if the media contents is an adverse media related to financial crime.
    The result will be 'True' if adverse media is found, and 'False' otherwise.

    Attributes:
        result: A boolean indicating if the media content is an adverse media or not.
    """

    result: bool = Field(
        description="""You are a helpful assistant to determine if the news mentions the search object (input keyword) that is related to financial crime.
            Please only return as format boolean (True/False) and determine whether the news related to financial crime as per the below criteria committed by the search object (input keyword).

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
            """ )

model = ChatOpenAI(model="gpt-4o", temperature=0) 
structured_model = model.with_structured_output(IsAdverseMedia)

system = """You are a helpful assistant to check if the contents contains adverse media related to financial crime, please return boolean: True. If the news is not related to financial crime, please return boolean: False"""

class SortingBy(str, Enum):
    """Represent different sorting logics for media.

    Use the enum to categorize sorting logics for media by "newest" or "relevance". 

    Attributes:
        NEWEST: The search criteria to sort media by the latest to the oldest.
        RELEVANCY: The search criteria to sort media by the most relevant to the least relevant. 
    """
    NEWEST = "newest"
    RELEVANCY = "relevance"

def get_cnn_news(
    keyword: str,
    sort_by: SortingBy,
    page
) -> t.Iterable[NewsArticle]:
    """Retrieve NewsArticle Objects related to financial crime.
    
    Yield each NewsArticle Object based on CNN's official website and that are related to financial crime one at a time, allowing iteration over each NewsArticle Object.
    
    Args:
        keyword(str): The keyword used to search CNN news.
        amount(int): The number of articles to be yielded.
        sort_by(SortingBy): The search critera to sort media by "newest" or "relevance". If SortingBy.NEWEST: Sort media by the latest to the oldest.
                            If SortingBy.RELEVANCY: Sort media by the most relevant to the least relevant. 
    """
    # count = 0
    # page = 1
    size_per_page = 3
    # news_article_collection = []
    # while count < amount:
    web = requests.get(
        'https://search.prod.di.api.cnn.io/content', 
        params={
            'q': keyword,
            'size': size_per_page,
            'sort': sort_by,
            'from': (page-1)*3,
            'page':page,
            'request_id':'stellar-search-19c44161-fd1e-4aff-8957-6316363aaa0e',
            'site':'cnn'
        }
    ) 
    news_collection = web.json().get("result")
    for i , news in enumerate(news_collection):
        if news["type"] == "VideoObject":
            continue
        url = news["path"]          
        article = NewsPlease.from_url(url)
        # news_article_collection.append(article)
        # return news_article_collection
        yield article


def handle_cnn_news(article:NewsArticle, count)-> t.Iterable[NewsArticle]:
    print(article)
    is_adverse_media= structured_model.invoke([SystemMessage(content=system)]+[HumanMessage(content=article.get_serializable_dict()['maintext'])])
    if is_adverse_media.result==True:
        print(is_adverse_media.result)
        count+=1
        print(count)
        yield article, count 
    else:
        print(is_adverse_media.result)
        return 

# def handle_cnn_news(news_article: t.List[NewsArticle], count)-> t.Iterable[NewsArticle]:

    # def is_adverse_media (article):
    #     is_adverse_media= structured_model.invoke([SystemMessage(content=system)]+[HumanMessage(content=article.get_serializable_dict()['maintext'])])
    #     return is_adverse_media.result
        
    # filtered_news_collection = list(filter(is_adverse_media, news_article_collection))
    # print(filtered_news_collection)
    # count+= len(filtered_news_collection)
    
    # return filtered_news_collection, count


# def handle_crime ():
#             news_summary= structured_model.invoke([SystemMessage(content=system)]+[HumanMessage(content=news_article.get_serializable_dict()['maintext'])]+[HumanMessage(content=news_article.get_serializable_dict()['date_publish'])])
#             news_summary: t.NewsSummary
#             if news_summary.is_adverse_news==True:
#                  yield news_article, news_summary.crimes
#                  count +=1                
#             # if count>=amount:
#             #     break
#         # page += 1


def cnn_news_pipeline(
    keyword: str,
    amount: int,
    sort_by: SortingBy,)-> t.Iterable[NewsArticle]:
    
    count = 0
    page = 0
    news_article_collection = []
    while count<amount:
        news_article = get_cnn_news(keyword, sort_by, page)
        # news_article_collection.append(get_cnn_news(keyword, sort_by, page))
        # print(news_article_collection)
        for news in news_article:
            if handle_cnn_news(news, count) is not None:
                filtered_news_and_count= handle_cnn_news(news, count)
                for f in filtered_news_and_count:
                    print(f)
                    filtered_news=f[0] 
                    count = f[1]
                    news_article_collection.append(filtered_news)
                    # print(f[0])
                    # print(f[1])
                    # raise ValueError
        
        # downloded_news, count= handle_cnn_news(news_article_collection, count)
        # print(downloded_news)
        # print(count)
                    if count>=amount:
                        break
        page+=1
    
    return news_article_collection
    


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("keyword", help="The keyword to search on CNN", type=str)
    parser.add_argument("amount", help="The amount of the crawled articles", type=int)
    parser.add_argument("-s", "--sort-by", help="The factor of news ranking", default=SortingBy.RELEVANCY)
    args = parser.parse_args()

    downloaded_news = cnn_news_pipeline(args.keyword, args.amount, args.sort_by)
    for news in downloaded_news:
        print(news)

    raise ValueError

    db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
    cur=db.cursor()

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
    
    client = QdrantClient(url="http://localhost:6333")
    client.create_collection(
        collection_name="cnn_news_chunk_vectors",
        vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
    )

    for news_article in downloaded_news:
        cur.execute(
                """INSERT INTO my_database.CNN_NEWS (title, keyword, description, maintext, date_publish, url)
                VALUES (%s, %s, %s, %s, %s, %s)""",
                (news_article.title,
                args.keyword, 
                news_article.description, 
                news_article.maintext, 
                news_article.date_publish,
                news_article.url)
                )
        article_id = cur.lastrowid
        db.commit()

        insert_chunk_table(news_article, article_id)

    cur.execute("select * from my_database.CNN_NEWS")
    for row in cur.fetchall():
        print(row)
    cur.close()
    db.close()


    
    