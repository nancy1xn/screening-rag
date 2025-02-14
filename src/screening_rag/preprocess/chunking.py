import typing as t
from langchain.text_splitter import(
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
import MySQLdb

db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
cur=db.cursor()
# cur.execute("DROP TABLE CHUNK_CNN_NEWS;")
# cur.execute("""
#     CREATE TABLE CHUNK_CNN_NEWS (
#         PK_CHUNK BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
#         text VARCHAR(1000),
#         start_position INT,
#         end_position INT,
#         Article_PK BIGINT UNSIGNED NOT NULL,
#         PRIMARY KEY(PK_CHUNK),
#         FOREIGN KEY (Article_PK) REFERENCES CNN_NEWS(Article_PK)
#     );
# """)

def chunk_text(text:str) -> t.Iterable[str]:
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True
    )
    text_split = character_splitter.split_text(text)
    token_splitter = SentenceTransformersTokenTextSplitter()

    for section in text_split:
        #  chunks.extend(token_splitter.split_text(section))
        yield token_splitter.split_text(section)

db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
cur=db.cursor()
cur.execute("select maintext, Article_PK from my_database.CNN_NEWS")
for row in cur.fetchall():
    row: t.Tuple
    maintext, article_pk = row
    chunks_example = chunk_text(maintext)
    for chunk in chunks_example:
            cur.execute(
                """INSERT INTO my_database.CHUNK_CNN_NEWS (text, Article_PK)
                VALUES (%s, %s)""",
                (chunk,
                article_pk)
             )
db.commit()
cur.execute("select * from my_database.CHUNK_CNN_NEWS")
for row in cur.fetchall():
    print(row)


    
