import os
import typing as t

import MySQLdb
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_text_splitters.sentence_transformers import (
    SentenceTransformersTokenTextSplitter,
)


def chunk_text(maintext: str) -> t.Iterable[str]:
    """Split text into several chunks which are smaller units

    Recursively split the text by characters, and then use a sentence model tokenizer to further split the text into chunk tokens.
    Chunks are yielded one group at a time ,allowing iteration over each chunks group.

    Args:
        text(str): The text to be split into chunks.
    """
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=256, chunk_overlap=50, add_start_index=True
    )
    text_splits = character_splitter.split_text(maintext)
    token_splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=256)

    for section in text_splits:
        yield token_splitter.split_text(section)


def insert_chunk_table(article_id, chunks):
    mysqldb_pw = os.getenv("MYSQLDB_PW")
    db = MySQLdb.connect(
        host="127.0.0.1", user="root", password=mysqldb_pw, database="my_database"
    )
    cur = db.cursor()
    # chunks = chunk_text(news_article.maintext)
    inserted_chunks = []
    for chunk in chunks:
        cur.execute(
            """INSERT INTO my_database.CHUNK_CNN_NEWS (text, parent_article_id)
                VALUES (%s, %s)""",
            (chunk, article_id),
        )
        chunk_id = cur.lastrowid
        inserted_chunks.append((chunk, article_id, chunk_id))

    db.commit()
    cur.execute("select * from my_database.CHUNK_CNN_NEWS")
    for row in cur.fetchall():
        print(row)
    cur.close()
    db.close()
    return inserted_chunks


if __name__ == "__main__":
    insert_chunk_table()
