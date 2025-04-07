import os
import re
import typing as t
from typing import List

import more_itertools as mit
import MySQLdb
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient


class SubquestionRelatedChunks:
    sub_question: int
    original_question: str
    text_collection: List[str]

    def __init__(self, sub_question, original_question, text_collection):
        self.sub_question = sub_question
        self.original_question = original_question
        self.text_collection = text_collection


class Relevance(BaseModel):
    """Assign a relevance score based on the relevance between the answer and the quesion.

    Define the guidelines for assinging a relevance score based on how much useful content the answer contains in relation to the corresponding original question.
    A score of 0 indicates low relevance, while 1 indicates high relevance.

    Attributes:
            result: A float between 0 and 1 represents how much the content of the answer is relevant to the corresponding question.
    """

    score: float = Field(
        description="""
            Please assign a relevance score based on how much useful content the answer contains in relation to the corresponding original question. 
            A score of 0 indicates low relevance, while 1 indicates high relevance. 
            A score below 0.5 indicates that the answer lacks sufficient valuable content and may be disregarded, 
            while a score of 0.5 or higher suggests the answer contains enough relevant information to be considered
        """
    )


def gen_report1(keyword: str) -> t.Dict[str, List[str]]:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
    Qdrant_domain = os.getenv("QDRANT_DOMAIN")
    client = QdrantClient(url=Qdrant_domain)
    subject = keyword

    original_question = [
        [
            f"q1_1 When was the company {subject} founded?",
            f"q1_2 Which country is the company {subject} headquartered in?",
            f"q1_3 What is the stock ticker of {subject} or its listing status? Please provide only relevant details.",
            f"q1_4 What type of business does the company {subject} provide?",
        ],
        [
            f"""Has the company {subject} been accused of committing any financial crimes? 
                If so, please provide the summary of financial crime the company {subject} accused of committing"""
        ],
    ]

    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_model = model.with_structured_output(Relevance)
    system = """
        You are a helpful assistant to provide relevence score based on how much useful content the answer contains in relation to the original question. 
        The score is within the range of 0 to 1.
    """

    def is_relevance_higher_than_threshold(qa_data: t.Tuple[str]):
        sub_question, searched_answer, _ = qa_data
        relevence_score_open_ai = structured_model.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=str(sub_question)),
                HumanMessage(content=str(searched_answer)),
            ]
        )
        return relevence_score_open_ai.score >= 0.5

    saved_chunks_group_1 = []
    for sub_question_index, question_value in enumerate(original_question[0]):
        question_openai_vectors = embeddings.embed_documents([question_value])
        question_openai_vectors: t.List[List[float]]
        search_results = client.query_points(
            collection_name="cnn_news_chunk_vectors",
            query=question_openai_vectors[0],
            limit=3,
        )

        related_subset = []
        for search_result in search_results.points:
            related_subset.append(
                (
                    question_value,
                    search_result.payload["text"],
                    search_result.payload["article_id"],
                )
            )
        filtered_qa_results = list(
            filter(
                is_relevance_higher_than_threshold,
                related_subset,
            )
        )
        saved_chunks_group_1.append(
            SubquestionRelatedChunks(
                sub_question=sub_question_index,
                original_question=original_question[0][sub_question_index],
                text_collection=filtered_qa_results,
            )
        )

    class ChatReport(BaseModel):
        result: str = Field(
            description="""
                (1)Help generate the final answer in relation to the corresponding original question according to the chunks materials from "text" collection. 
                (2)Include the corresponding [article_id] at the end of each sentence to indicate the source of the chunk. 
                * ONLY use information from the provided context chunks (with article_id).
                * DO NOT use any outside knowledge, even if the answer seems obvious.
                * DO NOT guess, generalize, or fabricate any part of the answer.
                * If there is no relevant information in the chunks for a question, reply exactly: → "No relevant information found."
                (3)Please refer to the examples below when generating the answers:
                
                        Question: 'q1_1 When was the company Google founded?'
                        Answer: 'Google was officially founded in 1998. [article_id:1]'

                        Question:'q1_2 Which country is the company Google headquartered in?'
                        Answer: 'Google's headquarters is located in Mountain View, California, USA.[article_id:2]' 

                        Question:'q1_3 What is the stock ticker of Google or its listing status? Please provide only relevant details. If it's not a public listed company, please answer 'Google is not a public listed company.'
                        Answer: 'Google's parent company, Alphabet Inc., is listed on the stock market. The stock ticker for Alphabet is GOOGL & GOOG traded on the NASDAQ exchange in USA [article_id:3]'

                        Question:'q1_4 What type of business does the company Google provide?'
                        Answer: 'Google is a tech company that primarily focuses on online services, advertising, cloud computing, and hardware, while also venturing into various other sectors.[article_id:4]
            """
        )

    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_model = model.with_structured_output(ChatReport)
    system_prompt = """You are a helpful assistant to generate the final answer in relation to the corresponding original question according to the chunks materials from "text" collection."""

    saved_answers = []

    MySQLdb_pw = os.getenv("MYSQLDB_PW")
    db = MySQLdb.connect(
        host="127.0.0.1", user="root", password=MySQLdb_pw, database="my_database"
    )
    cur = db.cursor()

    for subquestion_pair in saved_chunks_group_1:
        if subquestion_pair.text_collection == []:
            saved_answers.append(
                {
                    "sub_question": subquestion_pair.sub_question,
                    "final_answer": "No relevant information found.",
                }
            )
        else:
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

    final_answers_1 = []
    final_appendix_1 = []
    saved_final_answers = []
    unanswered_questions = []

    for ans in saved_answers:
        if ans["final_answer"] == "No relevant information found.":
            unanswered_questions.append(original_question[0][ans["sub_question"]])
        else:
            final_answers_1.append(ans["final_answer"])
            match = re.findall(r"\[article_id:(\d+)\]", str(ans))
            if match:
                for article_id in match:
                    query = (
                        "select ID, title, url from my_database.CNN_NEWS where ID = %s"
                    )
                    cur.execute(query, (int(article_id),))
                    # final_appendix_1.append(cur.fetchall())
                    final_appendix_1.append(mit.one(cur.fetchall()))
    set_appendix_1 = set(final_appendix_1)
    sorted_appendix_1 = sorted(set_appendix_1, key=lambda x: x[0])

    saved_final_answers.append(
        {
            "Client Background": final_answers_1,
            "Appendix of Client Background": sorted_appendix_1,
            "Unanswered Questions": unanswered_questions,
        }
    )

    print(saved_final_answers)
    return final_answers_1, sorted_appendix_1


if __name__ == "__main__":
    gen_report1("JP Morgan")
