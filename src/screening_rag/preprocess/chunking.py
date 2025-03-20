import typing as t
from langchain.text_splitter import(
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
import MySQLdb
from newsplease.NewsArticle import NewsArticle
from screening_rag.preprocess.chunking_qdrant import process_and_insert_chunks_to_cnn_news_chunk_vectors

def insert_chunk_table(news_article:NewsArticle, article_id):
    db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
    cur=db.cursor()
    # cur.execute("DROP TABLE CHUNK_CNN_NEWS;")
    # cur.execute("""
    #     CREATE TABLE CHUNK_CNN_NEWS (
    #         ID BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    #         text VARCHAR(1000),
    #         start_position INT UNSIGNED,
    #         end_position INT UNSIGNED,
    #         parent_article_id BIGINT UNSIGNED NOT NULL,
    #         PRIMARY KEY(ID),
    #         FOREIGN KEY (parent_article_id) REFERENCES CNN_NEWS(ID)
    #     );
    # """)

    def chunk_text(text:str) -> t.Iterable[str]:
        """Split text into several chunks which are smaller units
        
        Recursively split the text by characters, and then use a sentence model tokenizer to further split the text into chunk tokens. 
        Chunks are yielded one group at a time ,allowing iteration over each chunks group.
        
        Args:
            text(str): The text to be split into chunks.
        """
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=256,
            chunk_overlap=50,
            add_start_index=True
        )
        text_split = character_splitter.split_text(text)
        token_splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=256)

        for section in text_split:
            #  chunks.extend(token_splitter.split_text(section))
            yield token_splitter.split_text(section)

    db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
    cur=db.cursor()
    # cur.execute("select maintext, ID from my_database.CNN_NEWS")
    # for row in cur.fetchall():
    #     row: t.Tuple

        # maintext, article_pk = row
    chunks = chunk_text(news_article.maintext)
    for chunk in chunks:
            cur.execute(
                """INSERT INTO my_database.CHUNK_CNN_NEWS (text, parent_article_id)
                VALUES (%s, %s)""",
                (chunk,
                article_id)
            )
            chunk_id = cur.lastrowid
            # print(chunk)
            # print(type(chunk))
            # raise ValueError
            process_and_insert_chunks_to_cnn_news_chunk_vectors(chunk, article_id, chunk_id)
    
    db.commit()
    cur.execute("select * from my_database.CHUNK_CNN_NEWS")
    for row in cur.fetchall():
        print(row)
    cur.close()
    db.close()

 
if __name__ == "__main__":
    insert_chunk_table()




    
