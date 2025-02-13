import json
from langchain.text_splitter import(
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models
import MySQLdb

# with open ("article_filter2.json", "r") as article_file:
#     news_article_collection = json.load(article_file)

db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
cur=db.cursor()
cur.execute("CREATE TABLE CHUNK_CNN_NEWS (PK_CHUNK BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, text VARCHAR(1000), start_position INT, end_position INT, Article_PK BIGINT UNSIGNED NOT NULL, PRIMARY KEY(PK_CHUNK), FOREIGN KEY (Article_PK) REFERENCES CNN_NEWS(Article_PK));")

#->list[str]
def chunk_text(text:str):
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True
    )
    text_split = character_splitter.split_text(text)

    token_splitter = SentenceTransformersTokenTextSplitter()
    chunks = []
    for section in text_split:
        # chunks.extend(token_splitter.split_text(section))
        yield token_splitter.split_text(section)

# news_article_collection_result_string = "\n".join(json.dumps(d) for d in news_article_collection)

db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
cur=db.cursor()
cur.execute("select maintext, Article_PK from my_database.CNN_NEWS")
for row in cur.fetchall():
    chunks_example = chunk_text(row[0])
    if not isinstance(row[1], int):
        print(f"警告: Article_PK 不是整數型態 - {row[1]}")
        continue
    # cur.execute("select PK from my_database.CNN_NEWS")
    # print(chunks_example)
    for chunk in chunks_example:
        # assert isinstance(row[1], int)
            cur.execute(
                """INSERT INTO my_database.CHUNK_CNN_NEWS (text, Article_PK)
                VALUES (%s, %s)""",
                (chunk,
                row[1])
             )
db.commit()
cur.execute("select * from my_database.CHUNK_CNN_NEWS")
for row in cur.fetchall():
    print(row)
    


# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-large",
#     dimensions=1024
# )

# openai_vectors = embeddings.embed_documents(chunks_example)
# #記得先看numpy


# for chunk in chunks_example:
#     cur.executemany(
#             """INSERT INTO my_database.CHUNK_CNN_NEWS (text, PK)
#             VALUES (%s, %s)""",
#             chunk,
#             PK
#             )
    
#     db.commit()
# cur.execute("select * from my_database.CHUNK_CNN_NEWS")
# for row in cur.fetchall():
#     print(row)


# client = QdrantClient(url="http://localhost:6333")

# client.create_collection(
#     collection_name="cnn_news",
#     vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
# )

# assert client.collection_exists(collection_name="cnn_news")

# client.upsert(
#     collection_name="cnn_news",
#         points=[
#             models.PointStruct(
#                 id=1,
#                 payload={
#                 "data": "news",
#                 },
#                 vector=openai_vectors[0]
#             ),
#         ],
# )
