import json
import re
import typing as t
from typing import List

import more_itertools as mit
import qdrant_client
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
    Settings,
    select_crime_events_grounding_data_from_db,
    select_distinct_subjects_from_db,
)

settings = Settings()


# Retrieve directly associated entity names (e.g., CEO)
def get_linked_entities(
    existing_subjects: t.List[tuple], subject: str
) -> List[SimilarSubjects]:
    """
    Retrieve names of entities that are directly associated with the subject,
    such as its key personnel (e.g., CEO), but not other unrelated or similar entities.
    """
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


def get_points_similar_to_embedding(
    query: str,
    collection_name: str,
    limit: int,
    embedding_model: t.Optional[str] = "text-embedding-3-large",
    dimentions: t.Optional[int] = 3072,
    score_threshold: t.Optional[float] = None,
    extra_qdrant_conditions: qdrant_client.conversions.common_types.Filter = None,
) -> QueryResponse:
    embedder = OpenAIEmbeddings(model=embedding_model, dimensions=dimentions)
    client = QdrantClient(url=settings.QDRANT_DOMAIN)
    embedding = mit.one(embedder.embed_documents([query]))
    embedding: t.List[List[float]]
    return client.query_points(
        collection_name=collection_name,
        query=embedding,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=extra_qdrant_conditions,
    )


def convert_search_results_to_question_related_chunks(
    related_crime_events: QueryResponse, query: str
) -> List[QuestionRelatedChunks]:
    saved_chunks_group = []
    for event in related_crime_events.points:
        saved_chunks_group.append(
            QuestionRelatedChunks(
                original_question=query,
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
    generated_similar_subjects = get_linked_entities(existing_subjects, subject)
    related_crime_events = get_points_similar_to_embedding(
        original_question,
        collection_name="crime_cnn_news_vectors",
        limit=10,
        score_threshold=0.41,
        extra_qdrant_conditions=models.Filter(
            must=[
                models.FieldCondition(
                    key="subjects",
                    match=models.MatchAny(any=generated_similar_subjects.names),
                )
            ]
        ),
    )
    saved_chunks_group = convert_search_results_to_question_related_chunks(
        related_crime_events, original_question
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


if __name__ == "__main__":
    generate_crime_events_report("JP Morgan")
