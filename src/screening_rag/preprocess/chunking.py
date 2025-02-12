import json
from langchain.text_splitter import(
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models


with open ("article_filter2.json", "r") as article_file:
    news_article_collection = json.load(article_file)

def chunk_text(text:str) ->list[str]:
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=500,
        chunk_overlap=50,
    )
    text_split = character_splitter.split_text(text)

    token_splitter = SentenceTransformersTokenTextSplitter()
    chunks = []
    for section in text_split:
        chunks.extend(token_splitter.split_text(section))
    return chunks

news_article_collection_result_string = "\n".join(json.dumps(d) for d in news_article_collection)
chunks_example = chunk_text(news_article_collection_result_string)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024
)

openai_vectors = embeddings.embed_documents(chunks_example)

client = QdrantClient(url="http://localhost:6333")

# client.create_collection(
#     collection_name="cnn_news",
#     vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
# )

# assert client.collection_exists(collection_name="cnn_news")

client.upsert(
    collection_name="cnn_news",
        points=[
            models.PointStruct(
                id=1,
                payload={
                "data": "news",
                },
                vector=openai_vectors[0]
            ),
        ],
)
