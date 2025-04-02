import json
import os
import re
import typing as t
from typing import List, Optional

import MySQLdb
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models


class SimilarSubjects(BaseModel):
    names: List[str] = Field(
        description="""
                    (1)Given a set of subject names, generate a list of alternative words that are partially similar to the keyword or name input by the user based on the input set of subjects name. 
                    Perform partial keyword matching to find relevant alternatives.
                    (2)Instead of generating using ChatGPT, simply choose a list of alternative words from the provided/input set of subject names.
                    (3)The final set shall includes original input keyword.
                    """
    )


class SubquestionRelatedChunks(BaseModel):
    original_question: Optional[str]
    crime_id: Optional[int]
    time: Optional[str]
    subjects: Optional[List[str]]
    summary: Optional[str]
    adverse_info_type: Optional[List[str]]
    violated_laws: Optional[str]
    enforcement_action: Optional[str]


class ChatReport(BaseModel):
    result: List[str] = Field(
        description="""
                    (1)As per each instance, help generate the final answer in relation to the corresponding original question according to the materials based on the 'time', 'subject', 'summary', 'violated_laws', and 'enforcement_action' field in each instance.
                    (2)Include the corresponding [id] at the end of each answer to indicate the source of the chunk  based on the 'crime_id' in the instance.
                    (3)Include the crime time in the format YYYYMM at the beginning, based on the 'time' field in the instance. 
                    (4)Help deduplicate the list of instances in json format (crime events) based on similar content, considering both the time and the event details. If there are two similar instances describe the same crime event with overlapping details, YOU MUST RETAIN THE NECESSARY NEWS BASED ON BELOW INSTRUCTIONS:
                       Instructions:
                        **Date Deduplication**: If the crime details are identical but the dates differ, keep only the latest record.  
                        **Content Merging**: If the crime details are similar but some records contain more detailed descriptions, merge the information and retain the most complete version.  
                        **Judgment Deduplication**: If multiple records have the same judgment outcome, keep only one instance and remove duplicates.  
                        **Judgment Updates**: If the judgment has changed (e.g., from an ongoing case to a final settlement), retain only the latest judgment.  
                        **Legal Type Separation**: If the records refer to different types of legal proceedings (e.g., civil vs. criminal), keep them separate instead of merging.  

                    (5)Please refer to the examples below when generating the answers                    
                            Question: 'Has the company Google been accused of committing any financial crimes? If so, please provide the summary of financial crime the company Google accused of committing.
                            Answer: [
                                    '202210 Google was accused of enabling Putin's terrorist financing crimes by benefiting financially from his operations. The bank have agreed to pay $499 million to settle the lawsuit. [id: 9]',

                                    '202308 Google violated anti-money laundering laws, failing to implement effective measures, and violating US economic sanctions. SEC has fined Google €2.42 billion for breaching US economic sanctions.[id: 100]'
                                    ]
                    """
    )


def gen_report(keyword: str) -> t.Dict[str, List[str]]:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
    Qdrant_domain = os.getenv("QDRANT_DOMAIN")
    client = QdrantClient(url=Qdrant_domain)
    subject = keyword
    original_question = [
        [
            f"q1_1 When was the company {subject} founded?",
            f"q1_2 Which country is the company {subject} headquartered in?",
            "q1_3 What is the stock ticker of {subject} or its listing status? Please provide only relevant details.",  # 改問題因為stock ticker score原本不準確, stock ticker可能還是得手動查
            f"q1_4 What type of business does the company {subject} provide?",
        ],
        [
            f"""Has the company {subject} been accused of committing any financial crimes? 
                If so, please provide the summary of financial crime the company {subject} accused of committing"""
        ],
    ]
    MySQLdb_pw = os.getenv("MYSQLDB_PW")
    db = MySQLdb.connect(
        host="127.0.0.1", user="root", password=MySQLdb_pw, database="my_database"
    )
    cur = db.cursor()
    cur.execute("SELECT DISTINCT subject FROM my_database.SUBJECT_CNN_NEWS")
    existing_subjects = cur.fetchall()
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    system_prompt_subjectkeywords = """You are a helpful assistant to perform partial keyword matching to find relevant alternatives partially similar to the keyword input by the user. Remember that original input keyword shall be included """

    generated_subjects = model.with_structured_output(SimilarSubjects).invoke(
        [
            SystemMessage(content=system_prompt_subjectkeywords),
            HumanMessage(content=str(existing_subjects)),
            HumanMessage(content=keyword),
        ]
    )
    question_openai_vectors_group_2 = embeddings.embed_documents(original_question[1])
    question_openai_vectors_group_2: t.List[List[float]]
    search_results_group_2 = client.query_points(
        collection_name="crime_cnn_news_vectors",
        query=question_openai_vectors_group_2[0],
        limit=2,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="subjects",
                    match=models.MatchAny(
                        any=generated_subjects.names,
                    ),
                )
            ]
        ),
    )
    saved_chunks_group_2 = []
    for crime in search_results_group_2.points:
        if crime.score >= 0.41:
            saved_chunks_group_2.append(
                SubquestionRelatedChunks(
                    original_question=original_question[1][0],
                    crime_id=crime.payload["id"],
                    time=crime.payload["time"],
                    subjects=crime.payload["subjects"],
                    summary=crime.payload["summary"],
                    adverse_info_type=crime.payload["adverse_info_type"],
                    violated_laws=crime.payload["violated_laws"],
                    enforcement_action=crime.payload["enforcement_action"],
                )
            )

    sorted_time_saved_chunks_group_2 = sorted(
        saved_chunks_group_2, key=lambda x: x.time, reverse=True
    )
    json_sorted_time_saved_chunks_group_2 = json.dumps(
        [chunk.model_dump() for chunk in sorted_time_saved_chunks_group_2], indent=4
    )

    structured_model = model.with_structured_output(ChatReport)
    system_ans = """You are a helpful assistant to generate the final answer in relation to the corresponding original question according 
                    to the materials based on the 'time', 'subjects', 'summary', 'violated_laws', and 'enforcement_action' field in the payload.
                    In addition, given a list of crime-related instances in json format, please help deduplicate the list of instances in json format (crime events) based on similar content, considering both the time and the event details.
                    YOU MUST RETAIN THE NECESSARY NEWS BASED ON INSTRUCTIONS.
                 """

    MySQLdb_pw = os.getenv("MYSQLDB_PW")
    db = MySQLdb.connect(
        host="127.0.0.1", user="root", password=MySQLdb_pw, database="my_database"
    )
    cur = db.cursor()

    aggregated_2nd_level = structured_model.invoke(
        [
            SystemMessage(content=system_ans),
            HumanMessage(content=json_sorted_time_saved_chunks_group_2),
        ]
    )

    final_appendix_2 = []
    saved_final_answers = []

    match = re.findall(r"\[id: (\d+)\]", str(aggregated_2nd_level.result))
    if match:
        for id in match:
            query = (
                "select ID, title, url from my_database.CRIME_CNN_NEWS where ID = %s"
            )
            cur.execute(query, (int(id),))
            for row in cur.fetchall():
                final_appendix_2.append(row)

    sorted_appendix_2 = sorted(final_appendix_2, key=lambda x: x[0])
    saved_final_answers.append(
        {
            "Adverse Information Report Headline": aggregated_2nd_level.result,
            "Appendix of Adverse Information Report Headline": sorted_appendix_2,
        }
    )

    print(saved_final_answers)
    return aggregated_2nd_level.result, sorted_appendix_2


if __name__ == "__main__":
    gen_report("JPMorgan")
