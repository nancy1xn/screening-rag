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
from qdrant_client.http.models.models import QueryResponse


class SimilarSubjects(BaseModel):
    names: List[str] = Field(
        description="""
                    (1)Given a set of subject names, generate a list of alternative words that are partially similar to the keyword or name input by the user based on the input set of subjects name. 
                    Perform partial keyword matching to find relevant alternatives.
                    (2)Instead of generating using ChatGPT, simply choose a list of alternative words from the provided/input set of subject names.
                    (3)***The final set MUST includes original input keyword-subject.
                    (4)***Exclude other banks or financial institution.（We aim to focus on topics that are internal to keyword-subject or directly connected to it. Therefore, we exclude other banks to ensure thematic consistency and data purity.）
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


def select_distinct_subjects_from_db(subject: str) -> t.List[tuple]:
    mysqldb_pw = os.getenv("MYSQLDB_PW")
    db = MySQLdb.connect(
        host="127.0.0.1", user="root", password=mysqldb_pw, database="my_database"
    )
    cur = db.cursor()
    cur.execute("SELECT DISTINCT subject FROM my_database.SUBJECT_CNN_NEWS")
    existing_subjects = cur.fetchall()
    cur.close()
    db.close()
    return existing_subjects


def get_similar_subjects(
    existing_subjects: t.List[tuple], subject: str
) -> List[SimilarSubjects]:
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    system_prompt_subjectkeywords = """You are a helpful assistant to perform partial keyword matching to find relevant alternatives partially similar to the keyword input by the user. Remember that original input keyword shall be included """

    generated_similar_subjects = model.with_structured_output(SimilarSubjects).invoke(
        [
            SystemMessage(content=system_prompt_subjectkeywords),
            HumanMessage(content=str(existing_subjects)),
            HumanMessage(content=subject),
        ]
    )
    return generated_similar_subjects


def search_vectors_similar_to_query_and_matching_similar_subjects(
    original_question: str, generated_similar_subjects: SimilarSubjects
) -> QueryResponse:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
    qdrant_domain = os.getenv("QDRANT_DOMAIN")
    client = QdrantClient(url=qdrant_domain)
    question_openai_vectors_group = embeddings.embed_documents(original_question)
    question_openai_vectors_group: t.List[List[float]]
    search_results_group = client.query_points(
        collection_name="crime_cnn_news_vectors",
        query=question_openai_vectors_group[0],
        limit=10,
        score_threshold=0.05,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="subjects",
                    match=models.MatchAny(any=generated_similar_subjects.names),
                )
            ]
        ),
    )
    return search_results_group


def convert_search_results_to_subquestion_chunks(
    search_result_group: QueryResponse, original_question
) -> List[SubquestionRelatedChunks]:
    saved_chunks_group = []
    for event in search_result_group.points:
        saved_chunks_group.append(
            SubquestionRelatedChunks(
                original_question=original_question,
                crime_id=event.payload["id"],
                time=event.payload["time"],
                subjects=event.payload["subjects"],
                summary=event.payload["summary"],
                adverse_info_type=event.payload["adverse_info_type"],
                violated_laws=event.payload["violated_laws"],
                enforcement_action=event.payload["enforcement_action"],
            )
        )
    return saved_chunks_group


# score_threshold=0.41,


def deduplicate_and_generate_answer(
    saved_chunks_group: List[SubquestionRelatedChunks],
) -> List[str]:
    sorted_time_saved_chunks_group = sorted(
        saved_chunks_group, key=lambda x: x.time, reverse=True
    )
    json_sorted_time_saved_chunks_group = json.dumps(
        [chunk.model_dump() for chunk in sorted_time_saved_chunks_group], indent=4
    )
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_model = model.with_structured_output(ChatReport)
    system_ans = """You are a helpful assistant to generate the final answer in relation to the corresponding original question according 
                    to the materials based on the 'time', 'subjects', 'summary', 'violated_laws', and 'enforcement_action' field in the payload.
                    In addition, given a list of crime-related instances in json format, please help deduplicate the list of instances in json format (crime events) based on similar content, considering both the time and the event details.
                    YOU MUST RETAIN THE NECESSARY NEWS BASED ON INSTRUCTIONS.
                 """

    aggregated_2nd_level = structured_model.invoke(
        [
            SystemMessage(content=system_ans),
            HumanMessage(content=json_sorted_time_saved_chunks_group),
        ]
    )
    return aggregated_2nd_level.result


def extract_ids_from_aggregated_results(
    aggregated_2nd_level_result: List[str],
) -> List[str]:
    mysqldb_pw = os.getenv("MYSQLDB_PW")
    db = MySQLdb.connect(
        host="127.0.0.1", user="root", password=mysqldb_pw, database="my_database"
    )
    cur = db.cursor()
    match_ids = re.findall(r"\[id: (\d+)\]", str(aggregated_2nd_level_result))
    cur.close()
    db.close()
    return match_ids


def select_grounding_datas_from_db(match_ids) -> List[tuple]:
    mysqldb_pw = os.getenv("MYSQLDB_PW")
    db = MySQLdb.connect(
        host="127.0.0.1", user="root", password=mysqldb_pw, database="my_database"
    )
    cur = db.cursor()
    final_appendix = []
    for id in match_ids:
        query = "select ID, title, url from my_database.CRIME_CNN_NEWS where ID = %s"
        cur.execute(query, (int(id),))
        for row in cur.fetchall():
            final_appendix.append(row)

    sorted_appendix = sorted(final_appendix, key=lambda x: x[0])
    return sorted_appendix


def gen_report_crime_events(subject: str) -> t.Dict[str, List[str]]:
    original_question = f"""Has the company {subject} been accused or alleged to have committed or facilitated any financial crimes, 
or failed to prevent such crimes? If so, please summarize the incidents involving {subject}."""
    existing_subjects = select_distinct_subjects_from_db(subject)
    generated_similar_subjects = get_similar_subjects(existing_subjects, subject)
    search_result_group = search_vectors_similar_to_query_and_matching_similar_subjects(
        original_question, generated_similar_subjects
    )
    # print(search_result_group)
    saved_chunks_group = convert_search_results_to_subquestion_chunks(
        search_result_group, original_question
    )
    aggregated_2nd_level_results = deduplicate_and_generate_answer(saved_chunks_group)
    match_ids = extract_ids_from_aggregated_results(aggregated_2nd_level_results)
    sorted_appendices = select_grounding_datas_from_db(match_ids)
    saved_final_answers = []
    saved_final_answers.append(
        {
            "Adverse Information Report Headline": aggregated_2nd_level_results,
            "Appendix of Adverse Information Report Headline": sorted_appendices,
        }
    )
    # print(saved_final_answers)
    return aggregated_2nd_level_results, sorted_appendices


if __name__ == "__main__":
    gen_report_crime_events("JPMorgan")
