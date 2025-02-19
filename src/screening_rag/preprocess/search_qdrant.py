import typing as t
from typing import List
import numpy as np
from langchain_openai import OpenAIEmbeddings
import json
from qdrant_client.models import ScoredPoint
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List
from qdrant_client import QdrantClient, models

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=3072
)
client = QdrantClient(url="http://localhost:6333")
subject = "Binance"
question = {
            "Q1 Subject Background": [
                f"q1_1 When was the company {subject} founded?",
                f"q1_2 Which country is the company {subject} headquartered in?",
                f"q1_3 What is the stock ticker of Binance or its listing status? Please provide only relevant details.", #改問題因為stock ticker score原本不準確, stock ticker可能還是得手動查
                f"q1_4 What type of business does the company {subject} provide?"
            ],
            "Q2 Adverse Information Report Headline (ordered by timeline)": [
                f"q2_1 Has the company {subject} committed any crimes?",
                f"q2_2 When did the company {subject} commit a crime (the specific date, month, year)?",
                f"q2_3 What type of crime did the company {subject} commit?"
            ]
}
# Define a pydantic model to enforce the output structure
class Relevance(BaseModel):
    score: float = Field(
                         description="""Please assign a relevance score based on how much useful content the answer contains in relation to the original question. 
                                        A score of 0 indicates low relevance, while 1 indicates high relevance. 
                                        A score below 5 indicates that the answer lacks sufficient valuable content and may be disregarded, 
                                        while a score of 5 or higher suggests the answer contains enough relevant information to be considered
                                        """
                    )

# Create an instance of the model and enforce the output structure
model = ChatOpenAI(model="gpt-4o", temperature=0) 
structured_model = model.with_structured_output(Relevance)
# Define the system prompt
system = """You are a helpful assistant to provide relevence score based on how much useful content the answer contains in relation to the original question. 
            The score is within the range of 0 to 1."""

saved_chunks = []
text_collection = []
for question_key, question_value in question.items():
    question_openai_vectors = embeddings.embed_documents(question_value)
    question_openai_vectors: t.List[List[float]]
    
# print(question_openai_vectors)
# print(type(question_openai_vectors))

# search
    for question_index, question_vector in enumerate(question_openai_vectors):
        search_results = client.query_points(
            collection_name="cnn_news_chunk_vectors",
            query=question_vector,
            limit=3
        )
        relevence_score_open_ai= structured_model.invoke([SystemMessage(content=system)]+[HumanMessage(content=str(search_results))])
        # print('search_results',search_results)

        # print(type(search_results))
        # print(relevence_score_open_ai)
        # raise ValueError
        
        ## for index_result, result in enumerate(search_results.points): #search_results有三個, result等於每一個scoredpoint
            
            # print('index_result', index_result)
            # print('single_result', result)
            # print('type of result', type(result))
            # print('single_result_text',result.payload["text"])

            ##text_collection.append(result.payload["text"])
       
        # print(text_collection)
        # raise ValueError
        #         print("Original question",question_value[question_index])
        #         print("Search Results:", search_results)
        #         print(relevence_score_open_ai)
        #         print("result0",result[0], "result1",result[1])        
        #         raise ValueError


        saved_chunks.append({
            "question": question_value[question_index],
            "text": search_results,
            # or result["payload"]["text"]
            "score": relevence_score_open_ai
            })
        

print(saved_chunks)


#search
# for question_vector in question_openai_vectors:
#     results = client.query_points(
#         collection_name="cnn_news_chunk_vectors",
#         query=question_vector,
        # query_filter=models.Filter(
        # must=[
        #     models.FieldCondition(
        #         key="text",
        #         match=models.MatchValue(
        #             value="Binance",
        #         ),
        #     )
        # ]
        # ),
    #     limit=5
    # )

# print(results)
