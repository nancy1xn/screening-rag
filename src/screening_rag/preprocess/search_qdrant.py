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
import re
import MySQLdb
import streamlit as st

 # Define a pydantic model to enforce the output structure
class Relevance(BaseModel):
    """Assign a relevance score based on the relevance between the answer and the quesion.

    Define the guidelines for assinging a relevance score based on how much useful content the answer contains in relation to the corresponding original question. 
    A score of 0 indicates low relevance, while 1 indicates high relevance. 

    Attributes:
            result: A float between 0 and 1 represents how much the content of the answer is relevant to the corresponding question. 
    """
    score: float = Field(
                        description="""Please assign a relevance score based on how much useful content the answer contains in relation to the corresponding original question. 
                                        A score of 0 indicates low relevance, while 1 indicates high relevance. 
                                        A score below 0.5 indicates that the answer lacks sufficient valuable content and may be disregarded, 
                                        while a score of 0.5 or higher suggests the answer contains enough relevant information to be considered
                                        """
                    )

class SubquestionRelatedChunks:
    sub_question:int
    original_question:str
    text_collection: List[str]

    def __init__(self, sub_question, original_question, text_collection):
        self.sub_question = sub_question
        self.original_question =original_question
        self.text_collection = text_collection


def gen_report(
    keyword:str    
) ->t.Dict[str, List[str]]:
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=3072
    )
    client = QdrantClient(url="http://localhost:6333")
    subject = keyword

    original_question = [
        [
            f"q1_1 When was the company {subject} founded?",
            f"q1_2 Which country is the company {subject} headquartered in?",
            f"q1_3 What is the stock ticker of Binance or its listing status? Please provide only relevant details.", #改問題因為stock ticker score原本不準確, stock ticker可能還是得手動查
            f"q1_4 What type of business does the company {subject} provide?",
        ],
        [ 
            f"""Has the company {subject} been accused of committing any financial crimes? 
                If so, please provide the summary of financial crime the company {subject} accused of committing"""
        ]
    ]

#Group2
    # print(original_question[1])
    # question_openai_vectors_group_2 = embeddings.embed_documents(original_question[1])
    # question_openai_vectors_group_2: t.List[List[float]]
    # search_results_group_2 = client.query_points(
    #            collection_name="summary_cnn_news_vectors",
    #            query=question_openai_vectors_group_2[0],
    #            limit=100
    #        )
    # for search_result_group_2 in search_results_group_2.points:
    #     print(search_result_group_2)

#Group1
    # Create an instance of the model and enforce the output structure
    model = ChatOpenAI(model="gpt-4o", temperature=0) 
    structured_model = model.with_structured_output(Relevance)

    # Define the system prompt
    system = """You are a helpful assistant to provide relevence score based on how much useful content the answer contains in relation to the original question. 
                The score is within the range of 0 to 1."""

    def pass_threshold(qa_data: t.Tuple[str]):
        sub_question, searched_answer, _ = qa_data
        relevence_score_open_ai= structured_model.invoke([
            SystemMessage(content=system),
            HumanMessage(content=str(sub_question)),
            HumanMessage(content=str(searched_answer)),
        ])
        return relevence_score_open_ai.score >= 0.5

    saved_chunks_group_1 = []
    for sub_question_index, question_value in enumerate(original_question[0]):
        question_openai_vectors = embeddings.embed_documents([question_value])
        question_openai_vectors: t.List[List[float]]
        search_results = client.query_points(
            collection_name="cnn_news_chunk_vectors",
            query=question_openai_vectors[0],
            limit=3
        )
        
        related_subset = []
        for search_result in search_results.points:
            related_subset.append((
                question_value,
                search_result.payload["text"],
                search_result.payload["article_id"],
            ))
        filtered_qa_results = list(filter(
            pass_threshold,
            related_subset,
        ))

        saved_chunks_group_1.append(SubquestionRelatedChunks(
            sub_question=sub_question_index,
            original_question= original_question[0][sub_question_index],
            text_collection=filtered_qa_results,
        ))
            
    # for chunk in saved_chunks:
    #     print(chunk.sub_question)
    #     print(chunk.original_question)
    #     print(chunk.text_collection)

    class ChatReport(BaseModel):
        result: str = Field(
            description="""
                        (1)Help generate the final answer in relation to the corresponding original question according to the chunks materials from "text" collection. 
                        (2)Include the corresponding [article_id] at the end of each sentence to indicate the source of the chunk.
                        (3)Please refer to the examples below when generating the answers:
                        
                                Question: 'q1_1 When was the company Google founded?'
                                Answer: 'Google was officially founded in 1998. [article_id:1]'

                                Question:'q1_2 Which country is the company Google headquartered in?'
                                Answer: 'Google's headquarters is located in Mountain View, California, USA.[article_id:2]' 

                                Question:'q1_3 What is the stock ticker of Google or its listing status? Please provide only relevant details. If it's not a public listed company, please answer 'Google is not a public listed company.'
                                Answer: 'Google's parent company, Alphabet Inc., is listed on the stock market. The stock ticker for Alphabet is GOOGL & GOOG traded on the NASDAQ exchange in USA [article_id:3]'

                                Question:'q1_4 What type of business does the company Google provide?'
                                Answer: 'Google is a tech company that primarily focuses on online services, advertising, cloud computing, and hardware, while also venturing into various other sectors.[article_id:4]
                        """)


    model = ChatOpenAI(model="gpt-4o", temperature=0) 
    structured_model = model.with_structured_output(ChatReport)
    system_ans = """You are a helpful assistant to generate the final answer in relation to the corresponding original question according to the chunks materials from "text" collection."""

    saved_answers = []

    db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
    cur=db.cursor()

    for subquestion in saved_chunks_group_1:
        aggregated_2nd_level= structured_model.invoke([
            SystemMessage(content=system_ans),
            HumanMessage(content=str(subquestion.text_collection)),
            HumanMessage(content=str(subquestion.original_question))
        ]) #把original_question+searched_chunks+score一起丟入
        print(aggregated_2nd_level)

        saved_answers.append({
            "sub_question": subquestion.sub_question,
            "final_answer": aggregated_2nd_level.result,
        })

    final_answers_1 = []
    final_appendix_1 =[]
    final_answers_2 = []
    final_appendix_2 =[]
    saved_final_answers =[]

    for ans in saved_answers:   
        final_answers_1.append(ans['final_answer'])

        match = re.findall(r'\[article_id:(\d+)\]', str(ans))
        if match:
            # print(f"article_id:{match}")
            for id in match:
                num=int(id)
                query = "select ID, title, url from my_database.CNN_NEWS where ID = %s"
                cur.execute(query, (num,))
                for row in cur.fetchall():
                    # print(row)
                    # if ans['main_question'] ==0:
                        final_appendix_1.append(row)
                    # elif ans['main_question'] ==1:
                    #     final_appendix_2.append(row)
        else:
            print("Not found article_id")
    
    saved_final_answers.append({"Client Background":final_answers_1, 
            "Appendix of Client Background":final_appendix_1,})
#             "Adverse Information Report Headline":final_answers_2, 
#             "Appendix of Adverse Information Report Headline":final_appendix_2}) 
    print(saved_final_answers)     
#     return saved_final_answers
    print("Client Background:", final_answers_1)
    print("Appendix of Client Background:", final_appendix_1)
#     # print("Adverse Information Report Headline:", final_answers_2)
#     # print("Appendix of Adverse Information Report Headline:", final_appendix_2)


gen_report("Binance")