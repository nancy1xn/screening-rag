import streamlit as st
from enum import Enum
import requests
import typing as t
from typing import Union
from newsplease.NewsArticle import NewsArticle
from newsplease import NewsPlease
import MySQLdb
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.load import dumps, loads
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from screening_rag.preprocess.crime import Crime
from screening_rag.preprocess import crime
from datetime import datetime

    

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
            description="Please list all financial crime events reported in the news and summarize them in response to the questions defined in the 'Class Crime' section.")

# Create an instance of the model and enforce the output structure
model = ChatOpenAI(model="gpt-4o", temperature=0) 
structured_model = model.with_structured_output(NewsSummary)

# Define the system prompt
system = """You are a helpful assistant to check if the contents contains adverse media related to financial crime and help summarize the event, 
            please return boolean: True. If the news is not related to financial crime, please return boolean: False.

            In addition, please list out all financial crimes in the news to summarize the financial crime in terms of the time (ONLY USE "news published date"((newsarticle.date_publish))）, the event, 
            the crime type, the direct object of wrongdoing, and the laws or regulations action."""

class SortingBy(str, Enum):
    """
    Represent different sorting logics for media.

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
) -> t.Iterable[Union[NewsArticle, t.List[Crime]]]:
    """
    Retrieve NewsArticle Objects related to financial crime.
    
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
    while True: #!!!
        web = requests.get(
            'https://search.prod.di.api.cnn.io/content', 
            params={
                'q': keyword,
                'size': size_per_page,
                'sort': sort_by,
                'from': (page-1)*size_per_page,
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
            news_article = NewsPlease.from_url(url)
            if news_article.date_publish > latesttime:
                news_summary= structured_model.invoke([SystemMessage(content=system)]+[HumanMessage(content=news_article.get_serializable_dict()['maintext'])]+[HumanMessage(content=news_article.get_serializable_dict()['date_publish'])])
                news_summary: NewsSummary
                if news_summary.is_adverse_news==True:
                    yield news_article, news_summary.crimes
                    count +=1              
            elif news_article.date_publish <= latesttime:
                return  #!!!
        page += 1


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument("keyword", help="The keyword to search on CNN", type=str)
    # parser.add_argument("amount", help="The amount of the crawled articles", type=int)
    parser.add_argument("-s", "--sort-by", help="The factor of news ranking", default=SortingBy.NEWEST)
    args = parser.parse_args()

    keywords = ["JP Morgan financial crime"]

    db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
    cur=db.cursor()

    for keyword in keywords:
        cur.execute("SELECT date_publish FROM my_database.CRIME_CNN_NEWS WHERE keyword = %s ORDER BY date_publish DESC LIMIT 1", (keyword,))
        latesttime = cur.fetchall()
        print(latesttime)
        downloaded_news = get_cnn_news(keyword, args.sort_by, datetime(2025, 3, 10, 21, 19, 8)) 

        for news_article, crimes in downloaded_news:
            for c in crimes:
                crime_adverse_info_type = ",".join(c.adverse_info_type)
                cur.execute(
                    """
                    INSERT INTO my_database.CRIME_CNN_NEWS (title, keyword, date_publish, time, summary, adverse_info_type, violated_laws, enforcement_action, url)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (news_article.title,
                    keyword,
                    news_article.date_publish, 
                    c.time,
                    c.summary,
                    crime_adverse_info_type,
                    c.violated_laws,
                    c.enforcement_action,
                    news_article.url),
                )
                c.id = cur.lastrowid #cursor.lastrowid 是 資料庫游標（cursor） 物件的一個屬性，用來 取得最後一次執行 INSERT 語句時，自動產生的主鍵 ID
                print(c.id)

                # print(crime.subjects)
                # print(type(crime.subjects))

                crime.insert_to_qdrant(c)

                for subject in c.subjects:
                    cur.execute(
                        """INSERT INTO my_database.SUBJECT_CNN_NEWS(subject, parent_crime_id) VALUES (%s, %s)""",
                        (subject, c.id),
                    )
                    db.commit()
            print(crimes)

    cur.execute("select * from my_database.CRIME_CNN_NEWS")
    for row in cur.fetchall():
          print(row)

    cur.execute("select * from my_database.SUBJECT_CNN_NEWS")
    for row in cur.fetchall():
         print(row)

    # for news_article, crimes in get_news:
    #     for crime in crimes:
    #         crime_subjects= ",".join(crime.subjects) if isinstance(crime.subjects, (list)) else crime.subjects
            # insert_vectors(crime_subjects)
            # try:
    # insert_vectors(1)
            # except Exception as e:
            #     print(f"insert_vectors 錯誤: {e}")