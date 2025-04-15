import json
import os
import re
import typing as t
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.models.models import QueryResponse

from screening_rag.custom_types import (
    QuestionRelatedChunks,
    SimilarSubjects,
    StructuredDataChatReport,
)
from screening_rag.db import (
    select_crime_events_grounding_data_from_db,
    select_distinct_subjects_from_db,
)


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


def convert_search_results_to_question_related_chunks(
    search_result_group: QueryResponse, original_question
) -> List[QuestionRelatedChunks]:
    saved_chunks_group = []
    for event in search_result_group.points:
        saved_chunks_group.append(
            QuestionRelatedChunks(
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


def deduplicate_and_generate_answer(
    saved_chunks_group: List[QuestionRelatedChunks],
) -> List[str]:
    sorted_time_saved_chunks_group = sorted(
        saved_chunks_group, key=lambda x: x.time, reverse=True
    )
    json_sorted_time_saved_chunks_group = json.dumps(
        [chunk.model_dump() for chunk in sorted_time_saved_chunks_group], indent=4
    )
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_model = model.with_structured_output(StructuredDataChatReport)
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
    match_ids = re.findall(r"\[id: (\d+)\]", str(aggregated_2nd_level_result))
    return match_ids


def generate_crime_events_report(subject: str) -> t.Dict[str, List[str]]:
    original_question = f"""Has the company {subject} been accused or alleged to have committed or facilitated any financial crimes, 
or failed to prevent such crimes? If so, please summarize the incidents involving {subject}."""
    existing_subjects = select_distinct_subjects_from_db(subject)
    generated_similar_subjects = get_similar_subjects(existing_subjects, subject)
    search_result_group = search_vectors_similar_to_query_and_matching_similar_subjects(
        original_question, generated_similar_subjects
    )
    saved_chunks_group = convert_search_results_to_question_related_chunks(
        search_result_group, original_question
    )
    aggregated_2nd_level_results = deduplicate_and_generate_answer(saved_chunks_group)
    match_ids = extract_ids_from_aggregated_results(aggregated_2nd_level_results)
    sorted_appendices = select_crime_events_grounding_data_from_db(match_ids)
    saved_final_answers = []
    saved_final_answers.append(
        {
            "Adverse Information Report Headline": aggregated_2nd_level_results,
            "Appendix of Adverse Information Report Headline": sorted_appendices,
        }
    )
    return aggregated_2nd_level_results, sorted_appendices
