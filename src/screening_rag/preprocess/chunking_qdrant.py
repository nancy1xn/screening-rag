import typing as t
from typing import List
import numpy as np
import MySQLdb
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models

def process_and_insert_chunks_to_cnn_news_chunk_vectors(chunk, article_id, chunk_id):
    # create collections
    # client = QdrantClient(url="http://localhost:6333")
    # client.create_collection(
    #     collection_name="cnn_news_chunk_vectors",
    #     vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
    # )

    # db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
    # cur=db.cursor()
    # cur.execute("select ID, text, parent_article_id from my_database.CHUNK_CNN_NEWS")
    #只embed text, 其他插入payload
    unique_id = 0
    # for row in cur.fetchall():
    #     row: t.Tuple
        # pk_chunk, text, article_pk = row
        # print(f'one of the row {row}')

    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=3072
    )
    text_openai_vectors = embeddings.embed_documents([chunk[0]])
    text_openai_vectors: t.List[List[float]]
    
    for each_text_openai_vectors in text_openai_vectors:
        print(f'chunk_id:{chunk_id}')
        print(f'text:{chunk}')
        print(f'article_id:{article_id}')
        client = QdrantClient(url="http://localhost:6333")
        client.upsert(
            collection_name="cnn_news_chunk_vectors",
                points=[
                    models.PointStruct(
                        id=chunk_id,
                        payload={
                        "unique_id": chunk_id,
                        "text": chunk,
                        "article_id": article_id
                        },
                        vector=each_text_openai_vectors
                    ),
                ],
        )
        unique_id+=1

if __name__ == "__main__":
    process_and_insert_chunks_to_cnn_news_chunk_vectors()


#CURL -L -X GET 'http://localhost:6333/collections/cnn_news_chunk_vectors/points/1'
#curl -X DELETE "http://localhost:6333/collections/cnn_news_chunk_vectors"