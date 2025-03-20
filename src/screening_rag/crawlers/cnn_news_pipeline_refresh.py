from enum import Enum
import requests
import typing as t
from newsplease.NewsArticle import NewsArticle
from newsplease import NewsPlease
import MySQLdb
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from screening_rag.preprocess.chunking import insert_chunk_table
from datetime import datetime

# Define a pydantic model to enforce the output structure
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

# Create an instance of the model and enforce the output structure
model = ChatOpenAI(model="gpt-4o", temperature=0) 
structured_model = model.with_structured_output(IsAdverseMedia)

# Define the system prompt
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
    latesttime
) -> t.Iterable[NewsArticle]:
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
    size_per_page = 3
    while True:
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

            if article.date_publish > latesttime:
                model = ChatOpenAI(model="gpt-4o", temperature=0) 
                structured_model = model.with_structured_output(IsAdverseMedia)
                is_adverse_media= structured_model.invoke([SystemMessage(content=system)]+[HumanMessage(content=article.get_serializable_dict()['maintext'])])
                if is_adverse_media.result==True:
                    yield article
                    count +=1  
            elif article.date_publish <= latesttime:
                return                     
        page += 1


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--sort-by", help="The factor of news ranking", default=SortingBy.NEWEST)
    args = parser.parse_args()

    keywords = ["JP Morgan financial crime"]

    db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
    cur=db.cursor()

    for keyword in keywords:
        cur.execute("""SELECT date_publish 
                    FROM my_database.CNN_NEWS 
                    WHERE keyword = %s 
                    ORDER BY date_publish 
                    DESC LIMIT 1""", 
                    (keyword,))
        latesttime = cur.fetchall()
        downloaded_news = get_cnn_news(keyword, args.sort_by, datetime(2025, 3, 10, 21, 19, 8)) 

        for news_article in downloaded_news:
            cur.execute(
                    """INSERT INTO my_database.CNN_NEWS (title, keyword, description, maintext, date_publish, url)
                    VALUES (%s, %s, %s, %s, %s, %s)""",
                    (news_article.title,
                    keyword, 
                    news_article.description, 
                    news_article.maintext, 
                    news_article.date_publish,
                    news_article.url)
                    )
            db.commit()
            article_id = cur.lastrowid
            insert_chunk_table(news_article, article_id)

    cur.execute("select * from my_database.CNN_NEWS")
    for row in cur.fetchall():
        print(row)
    
    cur.close()
    db.close()


    
    