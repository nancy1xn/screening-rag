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
from typing import Optional

class SubquestionRelatedChunks(BaseModel):
    original_question:Optional[str]
    crime_id:Optional[int]
    time:Optional[str]
    subjects:Optional[str]
    summary:Optional[str]
    adverse_info_type:Optional[str]
    violated_laws:Optional[str]
    enforcement_action:Optional[str]

    # def __init__(self, original_question, crime_id, time, subjects, summary, adverse_info_type, violated_laws, enforcement_action):
    #     self.original_question = original_question
    #     self.crime_id = crime_id
    #     self.time = time
    #     self.subjects = subjects
    #     self.summary = summary
    #     self.adverse_info_type = adverse_info_type
    #     self.violated_laws = violated_laws
    #     self.enforcement_action = enforcement_action

class ChatReport(BaseModel):
    result: str = Field(
        description="""
                    (1)As per each instance, help generate the final answer in relation to the corresponding original question according to the materials based on the 'time', 'subject', 'summary', 'violated_laws', and 'enforcement_action' field in each instance.
                    (2)Include the corresponding [id] at the end of each answer to indicate the source of the chunk  based on the 'crime_id' in the instance.
                    (3)Include the crime time in the format YYYYMM at the beginning, based on the 'time' field in the instance.
                    (4)Help deduplicate the list of instances (crime events) based on similar content, considering both the time and the event details.
                    (5)Please refer to the examples below when generating the answers:
                    
                            Question: 'Has the company {subject} been accused of committing any financial crimes? If so, please provide the summary of financial crime the company {subject} accused of committing.
                            Answer: '201708 Google was accused of violating anti-money laundering laws, 
                                    failing to implement effective measures, and violating US economic sanctions. 
                                    SEC has fined Google €2.42 billion for breaching US economic sanctions.[id: 100]'
                    """)

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

# # Testing
#     print(original_question[1])
#     question_openai_vectors_group_2 = embeddings.embed_documents(original_question[1])
#     question_openai_vectors_group_2: t.List[List[float]]
#     search_results_group_2 = client.query_points(
#                collection_name="test_goldman_sachs_vectors",
#                query=question_openai_vectors_group_2[0],
#                limit=100
#            )
#     for search_result_group_2 in search_results_group_2.points:
#         print(search_result_group_2)

# gen_report("Goldman Sachs")

# Group2
    # print(original_question[1])
    question_openai_vectors_group_2 = embeddings.embed_documents(original_question[1])
    question_openai_vectors_group_2: t.List[List[float]]
    search_results_group_2 = client.query_points(
               collection_name="summary_cnn_news_vectors",
               query=question_openai_vectors_group_2[0],
               limit=100
           )
    saved_chunks_group_2 = []
    for search_result_group_2 in search_results_group_2.points:
        # print(search_result_group_2)
        if search_result_group_2.score>=0.41 and subject in search_result_group_2.payload["subjects"]: #fuzzy search?
        # if subject in search_result_group_2.payload["subjects"]:
            saved_chunks_group_2.append(SubquestionRelatedChunks(
            original_question = original_question[1][0], 
            crime_id = search_result_group_2.payload["id"], 
            time = search_result_group_2.payload["time"], 
            subjects = search_result_group_2.payload["subjects"], 
            summary = search_result_group_2.payload["summary"], 
            adverse_info_type = search_result_group_2.payload["adverse_info_type"], 
            violated_laws = search_result_group_2.payload["violated_laws"], 
            enforcement_action = search_result_group_2.payload["enforcement_action"]
             ))
         
    sorted_time_saved_chunks_group_2 = sorted(saved_chunks_group_2, key=lambda x:x.time, reverse = True)
    json_sorted_time_saved_chunks_group_2 = json.dumps([chunk.model_dump() for chunk in sorted_time_saved_chunks_group_2], indent=4)
    # json_sorted_time_saved_chunks_group_2 = json.dumps(sorted_time_saved_chunks_group_2, indent=4) 

    model = ChatOpenAI(model="gpt-4o", temperature=0) 
    structured_model = model.with_structured_output(ChatReport)
    system_ans = """You are a helpful assistant to generate the final answer in relation to the corresponding original question according 
                    to the materials based on the 'time', 'subjects', 'summary', 'violated_laws', and 'enforcement_action' field in the payload."""

    saved_answers = []
    db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
    cur=db.cursor()

    # for chunk in sorted_time_saved_chunks_group_2:
    aggregated_2nd_level= structured_model.invoke([
        SystemMessage(content=system_ans),
        HumanMessage(content=json_sorted_time_saved_chunks_group_2),
        # HumanMessage(content=str(chunk.crime_id)),
        # HumanMessage(content=str(chunk.time)),
        # HumanMessage(content=str(chunk.subjects)),
        # HumanMessage(content=str(chunk.violated_laws)),
        # HumanMessage(content=str(chunk.summary)),
        # HumanMessage(content=str(chunk.enforcement_action))
    ]) #把original_question+searched_chunks+score一起丟入
    # print(aggregated_2nd_level)

    saved_answers.append(aggregated_2nd_level.result)
    final_answers_2 = []
    final_appendix_2 =[]
    saved_final_answers =[]

    # for ans in saved_answers:   
    final_answers_2.append(saved_answers[0])
    match = re.findall(r'\[id: (\d+)\]', str(final_answers_2))
    if match:
            for id in match:
                num=int(id)
                query = "select ID, title, url from my_database.SUMMARY_CNN_NEWS where ID = %s"
                cur.execute(query, (num,))
                for row in cur.fetchall():
                    final_appendix_2.append(row)
    
    saved_final_answers.append({
            "Adverse Information Report Headline":final_answers_2, 
            "Appendix of Adverse Information Report Headline":set(final_appendix_2)}) 
    print(saved_final_answers)

#     return saved_final_answers
    # print("Adverse Information Report Headline:", final_answers_2)
    # print("Appendix of Adverse Information Report Headline:", final_appendix_2)

gen_report("Binance")