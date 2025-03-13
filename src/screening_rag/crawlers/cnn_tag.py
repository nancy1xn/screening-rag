import streamlit as st
from enum import Enum
import requests
import json
import typing as t
from newsplease.NewsArticle import NewsArticle
from newsplease import NewsPlease
import MySQLdb
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.load import dumps, loads
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import date

class AdverseInfoType(str, Enum):
    Sanction = "Sanction"
    Money_Laundering_Terrorist_Financing = "Money Laundering/ Terrorist Financing"
    Fraud = "Fraud"
    Bribery_Corruption = "Bribery/ Corruption"
    Organized_Crime = "Organized Crime"
    Internal_Control_Failures = "Internal AML/CFT Control Failures"
    Other = "Other catergory of Adverse Information"
    

# Define a pydantic model to enforce the output structure
class Crime(BaseModel):
    time:str = Field(
        description="""With format YYYYMM/ When did the searched object commit a financial crime? (if applicable)
        If you can find the time when financial crime occurs, use that exact time of the crime mentioned in the news as the answer. 
        If there is no exact time of occurrence mentioned in the news, use "news published date" as the answer. 
        Please do not make up the answer if there is no relevent answer."""
    )
    summary:str = Field(
            description ="""Has the searched object been accused of committing any financial crimes? 
            If so, please provide the summary of of financial crime is the search objected accused of committing """
    )
    adverse_info_type: t.List[AdverseInfoType]
    subjects:t.List[str] = Field(description="Who are the direct subjects of the financial crimes (ex: for those subjects, what are the roles or positions linked to the search object)?")
    violated_laws:str = Field(description="Which laws or regulations are relevant to this financial crime accused of committing by searched object?")
    enforcement_action:str = Field(
                       description="""What is the actual law enforcement action such as charges, prosecution, fines, 
                       or conviction are relevant to this financial crime accused of committing by searched object?"""
    )

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

            In addition, please list out all financial crimes in the news to summarize the financial crime in terms of the time, the event, 
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
    amount: int,
    sort_by: SortingBy,
) -> t.Iterable[NewsArticle]:
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
    while count < amount:
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
            news_summary= structured_model.invoke([SystemMessage(content=system)]+[HumanMessage(content=news_article.get_serializable_dict()['maintext'])])
            if news_summary.is_adverse_news==True:
                 yield news_article, news_summary.crimes
                 count +=1                
            if count>=amount:
                break
        page += 1


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("keyword", help="The keyword to search on CNN", type=str)
    parser.add_argument("amount", help="The amount of the crawled articles", type=int)
    parser.add_argument("-s", "--sort-by", help="The factor of news ranking", default=SortingBy.RELEVANCY)
    args = parser.parse_args()
    get_news = get_cnn_news(args.keyword, args.amount, args.sort_by)

    db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
    cur=db.cursor()

    cur.execute("DROP TABLE SUBJECT_CNN_NEWS;")
    cur.execute("DROP TABLE CRIME_CNN_NEWS ;")

    cur.execute("""CREATE TABLE CRIME_CNN_NEWS (
                 ID BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, 
                 title VARCHAR(300), 
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

    def insert_id_subject(subject, id):
        x=cur.execute(
            """INSERT INTO my_database.SUBJECT_CNN_NEWS(subject, parent_crime_id)
                VALUES (%s)""",
                (subject,
                 id)
                )
        return x

    for news_article, crimes in get_news:
        print(crimes)
        for crime in crimes:
            time, summary, adverse_info_type, subjects, violated_laws, enforcement_action = crime
            crime_adverse_info_type = ", ".join(crime.adverse_info_type) if isinstance(crime.adverse_info_type, (list, tuple)) else crime.adverse_info_type
            cur.execute(
                    """INSERT INTO my_database.CRIME_CNN_NEWS (title, time, summary, adverse_info_type, violated_laws, enforcement_action, url)
                     VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                     (news_article.title, 
                     crime.time,
                     crime.summary,
                     crime_adverse_info_type,
                     crime.violated_laws,
                     crime.enforcement_action,
                     news_article.url)
                     )
            foerign_key_ID = cur.lastrowid #cursor.lastrowid 是 資料庫游標（cursor） 物件的一個屬性，用來 取得最後一次執行 INSERT 語句時，自動產生的主鍵 ID
            for subject in crime.subjects:
                    cur.execute(
                        """INSERT INTO my_database.SUBJECT_CNN_NEWS(subject, parent_crime_id)
                        VALUES (%s, %s)""",
                            (subject, foerign_key_ID)
                            )
                    db.commit()

    cur.execute("select * from my_database.CRIME_CNN_NEWS")
    for row in cur.fetchall():
          print(row)

    cur.execute("select * from my_database.SUBJECT_CNN_NEWS")
    for row in cur.fetchall():
         print(row)