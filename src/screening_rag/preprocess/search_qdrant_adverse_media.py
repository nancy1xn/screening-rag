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
from json import JSONEncoder
from tomark import Tomark

class SimilarSubjects(BaseModel):
    names: List[str] = Field(
        description="""
                    (1)Given a set of subject names, generate a list of alternative words that are partially similar to the keyword or name input by the user based on the input set of subjects name. 
                    Perform partial keyword matching to find relevant alternatives.
                    (2)Instead of generating using ChatGPT, simply choose a list of alternative words from the provided/input set of subject names.
                    (3)The final set shall includes original input keyword.
                    """)

class SubquestionRelatedChunks(BaseModel):
    original_question:Optional[str]
    crime_id:Optional[int]
    time:Optional[str]
    subjects:Optional[List[str]]
    summary:Optional[str]
    adverse_info_type:Optional[List[str]]
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
    result: List[str] = Field(
        description="""
                    (1)As per each instance, help generate the final answer in relation to the corresponding original question according to the materials based on the 'time', 'subject', 'summary', 'violated_laws', and 'enforcement_action' field in each instance.
                    (2)Include the corresponding [id] at the end of each answer to indicate the source of the chunk  based on the 'crime_id' in the instance.
                    (3)Include the crime time in the format YYYYMM at the beginning, based on the 'time' field in the instance. THE ORDER of output list of entries SHOULD IGNORE TIMELINE BUT NEED TO BE BASED ON SIMILAR CONTENTS.
                    (4)Given a list of crime-related instances in json format. You MUST AGGREGATE AND SORT SIMILAR EVENTS based on event details.

                    Instructions:
                    -Group similar events: If multiple instances describe the same crime event with overlapping details, you MUST merge them into a single entry.
                    -Preserve key details: Ensure that significant differences (such as legal outcomes, lawsuit claims, settlements, or new developments) are retained in the merged entry.
                    
                    (5)Please refer to the examples below when generating the answers 
                    (NOTE: below 202208-202210 examples all mentioned Google enable and have a close relationship with Putin, so those events shall be deemed as similar contents
                           below 202301-202308 examples all mentioned Google was accused of violating anti-money laundering laws and being sued by SEC, so those events shall be deemed as similar contents):
                    
                            Question: 'Has the company {subject} been accused of committing any financial crimes? If so, please provide the summary of financial crime the company {subject} accused of committing.
                            Answer: 
                                    '202210 Google was accused of enabling Putin's terrorist financing crimes by benefiting financially from his operations. The bank have agreed to pay $499 million to settle the lawsuit. [id: 9]
                                     202209 Google is being sued by the government for allegedly having a close relationship with Putin and ignoring red flags related to his accounts, which were allegedly used for terrorist financing. The prosecutors filed a lawsuit to Google for potentially profiting from these illegal acts. [id: 14]
                                     202108 Google  is accused of enabling and benefiting from Putin's terrorist financing crimes...[id: 13]'

                                    '202308 Google violated anti-money laundering laws, failing to implement effective measures, and violating US economic sanctions. SEC has fined Google €2.42 billion for breaching US economic sanctions.[id: 100]
                                     202301 Google was being accused of violating anti-money laundering regulations, SEC has sued Google for the illegal acts.[id:101]'
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
    #select from subject table in set format
    db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
    cur=db.cursor()
    cur.execute("SELECT DISTINCT subject FROM my_database.SUBJECT_CNN_NEWS")
    multiple_subjects_name_subset = cur.fetchall()
    model = ChatOpenAI(model="gpt-4o", temperature=0) 
    structured_model = model.with_structured_output(SimilarSubjects)
    system_prompt_subjectkeywords = """You are a helpful assistant to perform partial keyword matching to find relevant alternatives partially similar to the keyword input by the user. Remember that original input keyword shall be included """
    
    generated_subjects= structured_model.invoke([
                    SystemMessage(content=system_prompt_subjectkeywords),
                    HumanMessage(content=str(multiple_subjects_name_subset)),
                    HumanMessage(content=keyword)])
    # print(generated_subjects)
    # raise ValueError
    # print(original_question[1])
    question_openai_vectors_group_2 = embeddings.embed_documents(original_question[1])
    question_openai_vectors_group_2: t.List[List[float]]
    search_results_group_2 = client.query_points(
               collection_name="crime_cnn_news_vectors",
               query=question_openai_vectors_group_2[0],
               limit=2,
               query_filter = models.Filter(
                   must=[
                    models.FieldCondition(
                        key="subjects",
                        match=models.MatchAny(
                          any=generated_subjects.names,  
                        ),
                    )   
                   ] 
               )
    )
    saved_chunks_group_2 = []
    for search_result_group_2 in search_results_group_2.points:
        # print(search_result_group_2)
        if search_result_group_2.score>=0.41: #fuzzy search?
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

    structured_model = model.with_structured_output(ChatReport)
    system_ans = """You are a helpful assistant to generate the final answer in relation to the corresponding original question according 
                    to the materials based on the 'time', 'subjects', 'summary', 'violated_laws', and 'enforcement_action' field in the payload.
                    In addition, given a list of crime-related instances in json format. you MUST aggregate similar events and sort them based on event details. Please DO NOT SORT EVENTS BASED ON TIMELINE."""

    db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
    cur=db.cursor()

    aggregated_2nd_level= structured_model.invoke([
        SystemMessage(content=system_ans),
        HumanMessage(content=json_sorted_time_saved_chunks_group_2),

    ]) #把original_question+searched_chunks+score一起丟入
    # print(aggregated_2nd_level)
   

    final_appendix_2 =[]
    saved_final_answers =[]
    list_saved_final_answers =[]

    match = re.findall(r'\[id: (\d+)\]', str(aggregated_2nd_level.result))
    if match:
            for id in match:
                query = "select ID, title, url from my_database.CRIME_CNN_NEWS where ID = %s"
                cur.execute(query, (int(id),))
                for row in cur.fetchall():
                    final_appendix_2.append(row)
    
    saved_final_answers.append({
            "Adverse Information Report Headline":aggregated_2nd_level.result, 
            "Appendix of Adverse Information Report Headline":final_appendix_2}) 
    
    # list_saved_final_answers.append("Adverse Information Report Headline")        
    # list_saved_final_answers.append("\n") 
    # for gpt_result in aggregated_2nd_level.result:
    #     list_saved_final_answers.append(gpt_result ) 
    #     list_saved_final_answers.append("\n") 

    # list_saved_final_answers.append("Appendix of Adverse Information Report Headline")
    # list_saved_final_answers.append("\n")
    # for appendix_item in final_appendix_2:
    #     list_saved_final_answers.append(str(appendix_item))
    #     list_saved_final_answers.append("\n") 
    # print(list_saved_final_answers)
    print(saved_final_answers)

    # #listtomd
    # markdown_output = ""
    # for item in list_saved_final_answers:
    #         markdown_output += f"{item}\n"
    # return markdown_output

    #json content
    # json_saved_final_answers = json.dumps(saved_final_answers, indent=4)
    # json_with_empty_lines = "\n\n".join(json_saved_final_answers.splitlines())
    # markdown_output =f"```json\n{json_with_empty_lines}\n```"
    # return markdown_output
   
    # use tomark tool
    markdown = Tomark.table(saved_final_answers)
    print(markdown)
    return markdown

    ##create a md file
    # json_saved_final_answers = json.dumps(saved_final_answers, indent=4)
    # with open("saved_final_answers.md", "w") as f:
    #     f.write(json_saved_final_answers)
   
    # markdown_output=""
    # for title, final_answers in saved_final_answers[0].items():
    #     markdown_output+=f"{title}\n"
    #     if isinstance(final_answers, list):
    #         for ans in final_answers:
    #             markdown_output+=f"{ans}\n"
    #     elif isinstance(final_answers, set):
    #         for ans in final_answers:
    #             markdown_output+=f"{ans}\n"
    # print(markdown_output)
    # return st.markdown(markdown_output)
                            
#     return saved_final_answers
    # print("Adverse Information Report Headline:", final_answers_2)
    # print("Appendix of Adverse Information Report Headline:", final_appendix_2)

if __name__ == "__main__":  
    gen_report("JPMorgan")