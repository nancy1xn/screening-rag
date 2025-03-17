import typing as t
from enum import Enum
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models


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
    id:int = None

def insert_to_qdrant(crime: Crime):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=3072
    )

    client = QdrantClient(url="http://localhost:6333")
    crime_openai_vectors = embeddings.embed_documents([str(crime.summary)])
    crime_openai_vectors: t.List[t.List[float]]
    crime_openai_vector = crime_openai_vectors[0]
    client.upsert(
        collection_name="crime_cnn_news_vectors",
            points=[
                models.PointStruct(
                    id=crime.id,
                    payload={
                    "id": crime.id,
                    "time": crime.time,
                    "subjects": crime.subjects,
                    "summary": crime.summary,
                    "adverse_info_type": crime.adverse_info_type,
                    "violated_laws": crime.violated_laws,
                    "enforcement_action": crime.enforcement_action,
                    },
                    vector=crime_openai_vector,
                ),
            ],
    )


    #CURL -L -X GET 'http://localhost:6333/collections/crime_cnn_news_vectors/points/1'
    #curl -X DELETE "http://localhost:6333/collections/crime_cnn_news_vectors"