from enum import Enum
import requests
import json
import typing as t
from newsplease.NewsArticle import NewsArticle
from newsplease import NewsPlease
from math import ceil
import MySQLdb
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.load import dumps, loads
 
class SortingBy(str, Enum):
    NEWEST = "newest"
    RELEVANCY = "relevance"
    
def get_cnn_news(
    keyword: str,
    amount: int,
    sort_by: SortingBy,
) -> t.List[NewsArticle]:

    count = 0
    page = 1
    news_start_from = 0
    news_article_object_collection =[]
    size_per_page = 3
  

    while page <= ceil(amount / size_per_page) and count < amount:
        web = requests.get(
            'https://search.prod.di.api.cnn.io/content', 
            params={
                'q': keyword,
                'size': size_per_page,
                'sort': sort_by,
                'from': news_start_from,
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
            prompt_template = ChatPromptTemplate([
            ("user", """
            You are a helpful assistant to determine if the news mentions the search object (input keyword) that is related to financial crime.
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
            """),
            MessagesPlaceholder("msgs")
            ])

            llm = ChatOpenAI(model="o1-mini")
            chain = LLMChain(llm=llm,prompt=prompt_template,verbose=True)
            message = chain.invoke({"msgs":[HumanMessage(content=str(article.get_serializable_dict()['maintext']))]})
            if "False" in message.values():
                message.clear()
            elif "False." in message.values():
                message.clear()
            else:
                yield article.get_serializable_dict()
                # news_article_object_collection.append(article.get_serializable_dict())
            count +=1                
            # print(f'count{count}')
            if count>=amount:
                break

        # print(f'page{page} successfully processed')
        page += 1
        news_start_from += 3

    # with open("article_filter2.json", "w") as file:
    #     json.dump(news_article_object_collection, file, indent=4)

get_news = get_cnn_news('putin financial crime', 2, SortingBy.RELEVANCY)

# with open("article_filter2.json", "r") as file:
#     news_article_collection = json.load(file)

db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
cur=db.cursor()
# cur.execute("DROP TABLE CHUNK_CNN_NEWS;")
# cur.execute("DROP TABLE CNN_NEWS;")
cur.execute("CREATE TABLE CNN_NEWS (Article_PK BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, title VARCHAR(300), description VARCHAR(3000), maintext MEDIUMTEXT, date_publish DATETIME, url VARCHAR(300), PRIMARY KEY(Article_PK));")

# cur.execute("CREATE TABLE CNN_NEWS_TEST (title VARCHAR(300), description VARCHAR(3000), maintext MEDIUMTEXT, date_publish DATETIME, url VARCHAR(300) );")

# cur.execute("DESCRIBE CNN_NEWS;")
# myresult = cur.fetchall()
# for x in myresult:
#     print(x)

# for news_article in news_article_collection:

for news_article in get_news:
    cur.execute(
            """INSERT INTO my_database.CNN_NEWS (title, description, maintext, date_publish, url)
            VALUES (%s, %s, %s, %s, %s)""",
            (news_article["title"], 
             news_article["description"], 
             news_article["maintext"], 
             news_article["date_publish"],
             news_article["url"])
             )
    db.commit()
cur.execute("select * from my_database.CNN_NEWS")
for row in cur.fetchall():
    print(row)




