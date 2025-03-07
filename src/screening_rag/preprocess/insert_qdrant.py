import typing as t
from typing import List
import numpy as np
import MySQLdb
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=3072
)

# # create collections
from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="summary_cnn_news_vectors",
    vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
)

# assert client.collection_exists(collection_name="summary_cnn_news_vectors")
# collection_info = client.get_collection("summary_cnn_news_vectors")
# print(collection_info)

db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
cur=db.cursor()

cur.execute("select ID, time, summary, adverse_info_type, subject, violated_laws, enforcement_action from my_database.SUMMARY_CNN_NEWS")

#只embed text, 其他插入payload
for crime in cur.fetchall():
    crime: t.Tuple
    crime_id, time, summary, adverse_info_type, subject, violated_laws, enforcement_action = crime
    print(f'one of the row {crime}')
    crime_openai_vectors = embeddings.embed_documents([str(summary)])
    crime_openai_vectors: t.List[List[float]]
    crime = crime_openai_vectors[0]
    client.upsert(
        collection_name="summary_cnn_news_vectors",
            points=[
                models.PointStruct(
                    id=crime_id,
                    payload={
                    "id": crime_id,
                    "time": time,
                    "subject": subject,
                    "summary": summary,
                    "adverse_info_type": adverse_info_type,
                    "violated_laws": violated_laws,
                    "enforcement_action": enforcement_action,
                    },
                    vector=crime,
                ),
            ],
    )

        # print(openai_vectors)
        # print(type(openai_vectors))
        # d = np.array(openai_vectors)
        # print(d.shape)
        # raise ValueError



    #search
    # results = client.search(
    #     collection_name="summary_cnn_news_vectors",
    #     query_vector=openai_vectors[0],
    #     limit=10,
    #     with_vectors=True  # 設置為 True 以返回向量
    # )
    # print(results)
    #retrieve
    # retrieved_point = client.retrieve(
    #     collection_name="summary_cnn_news_vectors",
    #     ids=[1],
    # )
    # print(retrieved_point)


    # #get collection
    # collection_info = client.get_collection("testing")
    # print(collection_info)
    # results, next_page = client.scroll(
    #     collection_name="summary_cnn_news_vectors",
    #     limit=100,                # 每次返回的資料點數量
    #     offset=None,              # 可選的起始偏移量
    #     with_vectors=True,        # 設為 True 來檢索向量
    #     with_payload=True,        # 設為 True 來檢索 payload（附加資料)
    # )
    # print(results, next_page)

    #CURL -L -X GET 'http://localhost:6333/collections/summary_cnn_news_vectors/points/1'
    #curl -X DELETE "http://localhost:6333/collections/summary_cnn_news_vectors"