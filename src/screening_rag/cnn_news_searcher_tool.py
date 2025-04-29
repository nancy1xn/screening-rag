import re
import typing as t
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from screening_rag.cnn_crime_searcher_tool import get_points_similar_to_embedding
from screening_rag.custom_types import (
    ChunkBasedChatReport,
    Relevance,
    SubquestionRelatedChunks,
)
from screening_rag.db import Settings, select_background_grounding_data_from_db

settings = Settings()


def is_relevance_higher_than_threshold(qa_data: t.Tuple[str]):
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_model = model.with_structured_output(Relevance)
    system = """
        You are a helpful assistant to provide relevence score based on how much useful content the answer contains in relation to the original question. 
        The score is within the range of 0 to 1.
    """
    sub_question, searched_answer, _ = qa_data
    relevence_score_open_ai = structured_model.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=str(sub_question)),
            HumanMessage(content=str(searched_answer)),
        ]
    )
    return relevence_score_open_ai.score >= 0.5


def filter_subsets(related_subset: List[tuple]) -> List[tuple]:
    filtered_qa_results = list(
        filter(
            is_relevance_higher_than_threshold,
            related_subset,
        )
    )
    return filtered_qa_results


def convert_search_results_to_subquestion_related_chunks(
    filtered_qa_results: List[tuple],
    original_questions: List[str],
    sub_question_index: int,
    saved_chunks_group: List,
) -> List[SubquestionRelatedChunks]:
    saved_chunks_group.append(
        SubquestionRelatedChunks(
            sub_question=sub_question_index,
            original_question=original_questions[sub_question_index],
            text_collection=filtered_qa_results,
        )
    )
    return saved_chunks_group


def generate_answer(saved_chunks_group: List[SubquestionRelatedChunks]) -> List[str]:
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_model = model.with_structured_output(ChunkBasedChatReport)
    system_prompt = """You are a helpful assistant to generate the final answer in relation to the corresponding original question according to the chunks materials from "text" collection."""
    saved_answers = []
    for subquestion_pair in saved_chunks_group:
        if subquestion_pair.text_collection:
            aggregated_2nd_level = structured_model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=str(subquestion_pair.text_collection)),
                    HumanMessage(content=str(subquestion_pair.original_question)),
                ]
            )

            saved_answers.append(
                {
                    "sub_question": subquestion_pair.sub_question,
                    "final_answer": aggregated_2nd_level.result,
                }
            )
        else:
            required_info = [
                "founding time",
                "headquarter's location",
                "listing status",
                "type of business",
            ]
            saved_answers.append(
                {
                    "sub_question": subquestion_pair.sub_question,
                    "final_answer": f"No relevant information found for {required_info[subquestion_pair.sub_question]}",
                }
            )
    return saved_answers


def extract_ids_from_saved_answers(ans: dict):
    match_ids = re.findall(r"\[article_id:(\d+)\]", str(ans))
    return match_ids


def generate_background_report(subject: str) -> t.Dict[str, List[str]]:
    original_question = [
        f"q1_1 When was the company {subject} founded?",
        f"q1_2 Which country is the company {subject} headquartered in?",
        f"q1_3 What is the stock ticker of {subject} or its listing status? Please provide only relevant details.",
        f"q1_4 What type of business does the company {subject} provide?",
    ]
    saved_chunks_group = []

    for sub_question_index, question_value in enumerate(original_question):
        related_subset = []
        query_response = get_points_similar_to_embedding(
            question_value, collection_name="cnn_news_chunk_vectors", limit=3
        )

        related_subset = map(
            lambda p: (question_value, p.payload["text"], p.payload["article_id"]),
            query_response.points,
        )
        filtered_qa_results = filter_subsets(related_subset)
        saved_chunks_group = convert_search_results_to_subquestion_related_chunks(
            filtered_qa_results,
            original_question,
            sub_question_index,
            saved_chunks_group,
        )
    saved_answers = generate_answer(saved_chunks_group)

    final_appendix = []
    final_answers = []
    for ans in saved_answers:
        final_answers.append(ans["final_answer"])
        match_ids = extract_ids_from_saved_answers(ans)
        final_appendix = select_background_grounding_data_from_db(
            match_ids, final_appendix
        )

    set_appendix = set(final_appendix)
    sorted_appendix = sorted(set_appendix, key=lambda x: x[0])
    return final_answers, sorted_appendix


if __name__ == "__main__":
    generate_background_report("JP Morgan")
