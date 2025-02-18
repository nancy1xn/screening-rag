from enum import Enum
import requests
import json
import typing as t
from newsplease.NewsArticle import NewsArticle
from newsplease import NewsPlease
from math import ceil
import MySQLdb
   
def is_news_article(news):
    return news["type"] == "NewsArticle"

class SortingBy(Enum):
    Newest: 0
    Relevancy: 1
    
def get_cnn_news(
    keyword: str,
    amount: int,
    sort_by: SortingBy,
) -> t.List[NewsArticle]:
    

    page = 1
    news_start_from = 0
    news_article_object_collection =[]
    size_per_page = 3
    while page <= ceil(amount / size_per_page):
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
            url = news["path"]          
            article = NewsPlease.from_url(url)
            news_article_object_collection.append(article.get_serializable_dict())
            
        print(f'page{page} successfully processed')
        page += 1
        news_start_from += 3

    with open("article_filter2.json", "w") as file:
        json.dump(news_article_object_collection, file, indent=4)
       
        
get_cnn_news('putin', 2, 2)

with open("article_filter2.json", "r") as file:
    news_article_collection = json.load(file)

db=MySQLdb.connect(password="",database="my_db")
# cur=db.cursor()
   
# for news_article in news_article_collection:
#     cur.execute(
#             """INSERT INTO my_db.CNN_NEWS (title, description, maintext, date_download, date_publish, url)
#             VALUES (%s, %s, %s, %s, %s, %s)""",
#             (news_article["title"], 
#              news_article["description"], 
#              news_article["maintext"], 
#              news_article["date_download"], 
#              news_article["date_publish"],
#              news_article["url"])
#              )
#     db.commit()
#     cur.execute("select * from my_db.CNN_NEWS")
#     for row in cur.fetchall():
#         print(row)
        