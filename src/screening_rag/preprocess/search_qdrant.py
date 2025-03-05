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

def gen_report(
    keyword:str    
) ->t.Dict[str, List[str]]:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=3072
    )
    client = QdrantClient(url="http://localhost:6333")
    subject = keyword
    # original_question = {
    #             "Q1 Subject Background": [
    #                 f"q1_1 When was the company {subject} founded?",
    #                 f"q1_2 Which country is the company {subject} headquartered in?",
    #                 f"q1_3 What is the stock ticker of Binance or its listing status? Please provide only relevant details.", #改問題因為stock ticker score原本不準確, stock ticker可能還是得手動查
    #                 f"q1_4 What type of business does the company {subject} provide?",

    #             ],
    #             "Q2 Adverse Information Report Headline (ordered by timeline)": [
    #                 f"q2_1 Has the company {subject} been accused of committing any financial crimes?",
    #                 f"q2_2 When did the company {subject} commit a financial crime (the specific date, month, year)?",
    #                 f"q2_3 What type of financial crime is the company {subject} accused of committing?",
    #                 f"q2_4 Which laws or regulations are relevant to this financial crime accused of committing by {subject}?"
    #             ]
    # }

    original_question = {
                "Q1 Subject Background": [
                    f"q1_1 When was the company {subject} founded?",
                    f"q1_2 Which country is the company {subject} headquartered in?",
                    f"q1_3 What is the stock ticker of Binance or its listing status? Please provide only relevant details.", #改問題因為stock ticker score原本不準確, stock ticker可能還是得手動查
                    f"q1_4 What type of business does the company {subject} provide?",

                ],
                "Q2 Adverse Information Report Headline (ordered by timeline)": [ 
                    f"""
                    When did the company {subject} commit a financial crime (the specific date, month, year)?
                    Has the company {subject} been accused of committing any financial crimes? If so, please provide the type of of financial crime is the company {subject} accused of committing
                    Which laws or regulations are relevant to this financial crime accused of committing by {subject}?"""
                ]
    }

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

    # Create an instance of the model and enforce the output structure
    model = ChatOpenAI(model="gpt-4o", temperature=0) 
    structured_model = model.with_structured_output(Relevance)

    # Define the system prompt
    system = """You are a helpful assistant to provide relevence score based on how much useful content the answer contains in relation to the original question. 
                The score is within the range of 0 to 1."""

    saved_chunks = []

    for main_question_index, (question_key,question_value) in enumerate(original_question.items()):
        question_openai_vectors = embeddings.embed_documents(question_value)
        question_openai_vectors: t.List[List[float]]

    # search
        for question_index, question_vector in enumerate(question_openai_vectors):
            search_results = client.query_points(
                collection_name="cnn_news_chunk_vectors",
                query=question_vector,
                limit=18
            )
            relevence_score_open_ai= structured_model.invoke([SystemMessage(content=system)]+[HumanMessage(content=str(search_results))]+[HumanMessage(content=str(question_value[question_index]))])
            
            # print('original_question ',question_value[question_index])
            # print('search_results',search_results)
            # print(relevence_score_open_ai)

            text_collection = [] #注意每次三個蒐集完都要歸零不然會一直累積
            for result in search_results.points: #search results是QueryReponse type 要先用point取出attribute, search_results有三個, result等於每一個scoredpoint
                if relevence_score_open_ai.score >=0.5:
                    text_collection.append(result.payload["text"])
                    text_collection.append(f'[article_id:{result.payload["article_id"]}]') #加上article_id在每個文章後面
                else:
                    text_collection.append("None")

            class RelatedChunkCollection:
                main_question:int
                sub_question:int
                original_question:str
                text_collection: List[str]
                score= float

                def __init__(self, main_question, sub_question, original_question, text_collection, score):
                    self.main_question = main_question
                    self.sub_question = sub_question
                    self.original_question =original_question
                    self.text_collection = text_collection
                    self.score = score

            saved_chunks.append(RelatedChunkCollection(
                main_question=main_question_index,
                sub_question=question_index,
                original_question= question_value[question_index],
                text_collection=text_collection,
                score=relevence_score_open_ai.score,
            ))
            
    for chunk in saved_chunks:
        print(chunk.main_question)
        print(chunk.sub_question)
        print(chunk.original_question)
        print(chunk.text_collection)
        print(chunk.score)

    # class ChatReport(BaseModel):
    #     result: str = Field(
    #         description="""(1)Help generate the final answer in relation to the corresponding original question according to the chunks materials from "text" collection. 
    #                        (2)Include the corresponding [article id] at the end of each sentence to indicate the source of the chunk.

    #                        Please refer to the examples below when generating the answers:
    #                        Question: 'q1_1 When was the company Google founded?'
    #                        Answer: 'Google was officially founded in 1998. [article_id:1]'

    #                        Question:'q1_2 Which country is the company Google headquartered in?'
    #                        Answer: 'Google's headquarters is located in Mountain View, California, USA.[article_id:2]' 

    #                        Question:'q1_3 What is the stock ticker of Google or its listing status? Please provide only relevant details. If it's not a public listed company, please answer 'Google is not a public listed company.'
    #                        Answer: 'Google's parent company, Alphabet Inc., is listed on the stock market. The stock ticker for Alphabet is GOOGL & GOOG traded on the NASDAQ exchange in USA [article_id:3]'

    #                        Question:'q1_4 What type of business does the company Google provide?'
    #                        Answer: 'Google is a tech company that primarily focuses on online services, advertising, cloud computing, and hardware, while also venturing into various other sectors.[article_id:4]

    #                        Question:'q2_1 Has Google been accused of committing any financial crimes?'
    #                        Answer: 'Yes [article_id:5]'

    #                        Question:'q2_2 When did the company Google commit a financial crime (the specific date, month, year)?'
    #                        Answer: 'June 27, 2017 [article_id:5]'

    #                        Question:'q2_3 What type of financial crime is the company Google accused of committing?'
    #                        Answer:'Google has abused its market dominance as a search engine by giving an illegal advantage to another Google product, its comparison shopping service.[article_id:5]'

    #                        Question:'q2_4 Which laws or regulations are relevant to this financial crime accused of committing by Google?'
    #                        Answer:'The European Commission has fined Google €2.42 billion for breaching EU antitrust rules. [article_id:5]'   
    #     """)


    #  (3)In Question2, you will see excerpts from multiple articles, please use **every searched article based on article ID SEPERATELY** to answer subquestions in Question2 (only the excerpt that contains information relevant to the question), and DO NOT combine information from different excerpts. 
    #                           [article 1]:answer1
    #                           [article 2]:answer2


    # Please provide answers in the format below (MULIPLE ANSWERS ARE ALLOWED, PLEASE PROVIDE AS MUCH ANSWRS AS YOU CAN):
                                #Answer1: August 2017, Google was accused of violating anti-money laundering laws, failing to implement effective measures, and violating US economic sanctions. [article 1]
                                #Answer2: June 20, 2018, Google has been accused of engaging in anti-money laundering, unlicensed money transmitting, and sanctions violations.Google's founder is sentenced to 4 months in prison on money-laundering violations.[article 2]
                                #Answer3: June 27, 2017, Google has abused its market dominance as a search engine by giving an illegal advantage to another Google product, its comparison shopping service. The European Commission has fined Google €2.42 billion for breaching EU antitrust rules.[article 3]            
                            

    class ChatReport(BaseModel):
        result: str = Field(
            description="""(1)Help generate the final answer in relation to the corresponding original question according to the chunks materials from "text" collection. 
                        (2)Include the corresponding [article_id] at the end of each sentence to indicate the source of the chunk.
                        (3)In Question 2, you will see chunks from multiple corresponding[article_id]. Please follow these guidelines for answering:
                                **Each answer must strictly come from only one corresponeding [article_id].Do not mix information from different corresponding [article_id].**
                                1. **Use only the information relevant to the specific question** from each corresponding article. 
                                2. **DO NOT COMBINE INFORMATION FROM DIFFERENT CORRESPONDING [article_id] in EACH ANSWER, YET MULIPLE ANSWERS ARE ALLOWED**.
                                3. EACH ANSWER should be based SOLEY ON ONE SPECIFIC CORRESPONDING [article_id], not from multiple [article_id].
                                5. If the article does not contain relevant information, leave it blank (do not make up any answer).
                                6. Except for the blank questions, provide answers from the articles that contain the relevant information for the rest of the specific questions.
                                7. Please ensure that ALL articles from differnent [article_id] that contains relevant info are included in different answers, not just specific article_ids.
                                    
                        (4)Please refer to the examples below when generating the answers:
                        
                                Question: 'q1_1 When was the company Google founded?'
                                Answer: 'Google was officially founded in 1998. [article_id:1]'

                                Question:'q1_2 Which country is the company Google headquartered in?'
                                Answer: 'Google's headquarters is located in Mountain View, California, USA.[article_id:2]' 

                                Question:'q1_3 What is the stock ticker of Google or its listing status? Please provide only relevant details. If it's not a public listed company, please answer 'Google is not a public listed company.'
                                Answer: 'Google's parent company, Alphabet Inc., is listed on the stock market. The stock ticker for Alphabet is GOOGL & GOOG traded on the NASDAQ exchange in USA [article_id:3]'

                                Question:'q1_4 What type of business does the company Google provide?'
                                Answer: 'Google is a tech company that primarily focuses on online services, advertising, cloud computing, and hardware, while also venturing into various other sectors.[article_id:4]

                                Question2: 
                                '
                                When did the company Google commit a financial crime (**ONLY SHOW the specific date, month, year**)?
                                Has Google been accused of committing any financial crimes? If so, please provide the type of of financial crime is the company Google accused of committing.
                                Which laws or regulations are relevant to this financial crime accused of committing by Google?'
                                
                                Please provide concise answers in the format below: 
                                (The amount of the answers are NOT LIMITED TO 2. 
                                 **Each answer must strictly come from only one corresponding [article_id].Do not mix information from different sources.**
                                 **Please ensure that ALL articles from differnent [article_id] that contains relevant info are included in different answers**):
                                
                                Answer1:'June 27, 2017 [article_id:5]'
                                        'Google has abused its market dominance as a search engine by giving an illegal advantage to another Google product, its comparison shopping service.[article_id:5]'
                                        'The European Commission has fined Google €2.42 billion for breaching EU antitrust rules. [article_id:5]'   
                                Answer2:'June 20, 2018 [article_id:6]'
                                        'Google has been accused of engaging in anti-money laundering, unlicensed money transmitting, and sanctions violations.[article_id:6]'
                                        'Google's founder is sentenced to 4 months in prison on money-laundering violations. [article_id:6]' 
                """)

    model = ChatOpenAI(model="gpt-4o", temperature=0) 
    structured_model = model.with_structured_output(ChatReport)
    system_ans = """You are a helpful assistant to generate the final answer in relation to the corresponding original question according to the chunks materials from "text" collection."""

    saved_answers = []

    db=MySQLdb.connect(host="127.0.0.1", user = "root", password="my-secret-pw",database="my_database")
    cur=db.cursor()

    for chunk in saved_chunks:
        final_ans= structured_model.invoke([SystemMessage(content=system_ans)]+[HumanMessage(content=str(chunk.text_collection))]+[HumanMessage(content=str(chunk.original_question))]) #把original_question+searched_chunks+score一起丟入
        # print(final_ans)

        saved_answers.append({
                "main_question": chunk.main_question,
                "sub_question": chunk.sub_question,
                "final_answer": final_ans.result,
                })

    answers_dict_1 = []
    appendix_dict_1 =[]
    answers_dict_2 = []
    appendix_dict_2 =[]
    saved_final_answers =[]

    for ans in saved_answers:   
        # print(ans) 
        if ans['main_question'] ==0:
            answers_dict_1.append(ans['final_answer'])
        elif ans['main_question'] ==1:
            answers_dict_2.append(ans['final_answer'])

        match = re.findall(r'\[article_id:(\d+)\]', str(ans))
        if match:
            # print(f"article_id:{match}")
            for id in match:
                num=int(id)
                query = "select ID, title, url from my_database.CNN_NEWS where ID = %s"
                cur.execute(query, (num,))
                for row in cur.fetchall():
                    # print(row)
                    if ans['main_question'] ==0:
                        appendix_dict_1.append(row)
                    elif ans['main_question'] ==1:
                        appendix_dict_2.append(row)
        # else:
            # print("Not found article_id")
    
    saved_final_answers.append({"Client Background":answers_dict_1, 
            "Appendix of Client Background":appendix_dict_1, 
            "Adverse Information Report Headline":answers_dict_2, 
            "Appendix of Adverse Information Report Headline":appendix_dict_2}) 
    print(saved_final_answers)     
    return saved_final_answers
    # print("Client Background:", answers_dict_1)
    # print("Appendix of Client Background:", appendix_dict_1)
    # print("Adverse Information Report Headline:", answers_dict_2)
    # print("Appendix of Adverse Information Report Headline:", appendix_dict_2)


gen_report("Binance")