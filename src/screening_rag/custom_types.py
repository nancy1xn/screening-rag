import typing as t
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AdverseInfoType(str, Enum):
    Sanction = "Sanction"
    Money_Laundering_Terrorist_Financing = "Money Laundering/ Terrorist Financing"
    Fraud = "Fraud"
    Bribery_Corruption = "Bribery/ Corruption"
    Organized_Crime = "Organized Crime"
    Internal_Control_Failures = "Internal AML/CFT Control Failures"
    Other = "Other catergory of Adverse Information"


class Crime(BaseModel):
    time: str = Field(
        description="""
            You MUST use the news published date (newsarticle.date_publish) as the ONLY reference for determining the financial crime occurrence time. 
            The answer MUST be formatted as YYYYMM. Do NOT infer the date from any other clues in the text."""
    )
    summary: str = Field(
        description="""Has the searched object been accused of committing any financial crimes? 
            If so, please provide the summary of of financial crime is the search objected accused of committing """
    )
    adverse_info_type: t.List[AdverseInfoType]
    subjects: t.List[str] = Field(
        description="Who are the direct subjects of the financial crimes (ex: for those subjects, what are the roles or positions linked to the search object)?"
    )
    violated_laws: str = Field(
        description="Which laws or regulations are relevant to this financial crime accused of committing by searched object?"
    )
    enforcement_action: str = Field(
        description="""What is the actual law enforcement action such as charges, prosecution, fines, 
                       or conviction are relevant to this financial crime accused of committing by searched object?"""
    )
    id: int = None


class NewsSummary(BaseModel):
    """
    1.Check if the media content is considered as adverse media.
    2.List out and provide summary for all financial crimes in the news.

    -Define the guidelines for checking if the media contents is an adverse media related to financial crime.
     The result will be 'True' if adverse media is found, and 'False' otherwise.
    -List out all financial crimes in the news to summarize the financial crime in terms of the time, the event, the crime type, the direct object of wrongdoing, and the laws or regulations action.

    Attributes:
        is_adverse_news: A boolean indicating if the media content is an adverse media or not.
        crimes: list of 'Class Crime' indicating all financial crimes summary reported in the news.")

    """

    is_adverse_news: bool = Field(
        description="""Please determine if the news mentions the search object (input keyword) that is related to financial crime.
                    Please only return as format boolean (True/False) and determine whether the news related to financial crime accused of committing by the search object (input keyword) as per the below criteria:

                    (1) If the news is related to financial crime, please return boolean: True
                    (2) If the news is not related to financial crime, please return boolean: False

                    **Criteria for Financial Crime**:
                    Financial crime includes the following categories (Media coverage triggers: Regulatory Enforcement Actions, Convictions, Ongoing investigations, or allegations related to these):

                    1. Money laundering
                    2. Bribery and corruption
                    3. Fraud or weakness in fraud prevention controls
                    4. Stock exchange irregularities, insider trading, market manipulation
                    5. Accounting irregularities
                    6. Tax evasion and other tax crimes (related to direct and indirect taxes)
                    7. Regulatory enforcement actions against entities in the regulated sector with links to the client
                    8. Major sanctions breaches (including transactions with or any other involvement with sanctioned countries, territories, or countries under comprehensive sanction regimes)
                    9. Terrorism (including terrorist financing)
                    10. Illegal trade in narcotic drugs and psychotropic substances, weapons trafficking, or dealing in stolen goods.
                    11. Smuggling (including matters related to customs duties, excise taxes, and other applicable taxes)
                    12. Human trafficking or smuggling of migrants
                    13. Sexual exploitation of child labor
                    14. Extortion, counterfeiting, forgery, piracy of products
                    15. Organized crime or racketeering
                    16. Profiting from serious crimes (e.g., kidnapping, unlawful confinement, hostage-taking, robbery, theft, murder, or inflicting serious bodily harm).
                    17. Profiting from environmental crimes
                    18. Profiting from other unethical or criminal behavior

                    **Search Object**:
                    The search object refers to the keyword used to search for relevant news. In this case, it would be the term provided via a search request, for example:
                    `requests.get('https://search.prod.di.api.cnn.io/content', params={{'q': keyword}})`
                    """
    )

    crimes: t.List[Crime] = Field(
        description="Please list all financial crime events reported in the news and summarize them in response to the questions defined in the 'Class Crime' section."
    )


class SortingBy(str, Enum):
    """Represent different sorting logics for media.

    Use the enum to categorize sorting logics for media by "newest" or "relevance".

    Attributes:
        NEWEST: The search criteria to sort media by the latest to the oldest.
        RELEVANCY: The search criteria to sort media by the most relevant to the least relevant.
    """

    NEWEST = "newest"
    RELEVANCY = "relevance"


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


class ChunkBasedChatReport(BaseModel):
    result: str = Field(
        description="""
            (1)Help generate the final answer in relation to the corresponding original question according to the chunks materials from "text" collection. 
            (2)Include the corresponding [article_id] at the end of each sentence to indicate the source of the chunk. 
            * ONLY use information from the provided context chunks (with article_id).
            * DO NOT use any outside knowledge, even if the answer seems obvious.
            * DO NOT guess, generalize, or fabricate any part of the answer.
            * Answer the question based on the provided context. If the context contains clues or implications, make reasonable inferences. If information for any of these items is not available, please respond clearly with:
                -> "No relevant information found for founding time."
                -> "No relevant information found for headquarters location."
                -> "No relevant information found for listing status."
                -> "No relevant information found for type of business."
            (3)Please refer to the examples below when generating the answers:
            
                    Question: 'q1_1 When was the company Google founded?'
                    Answer: 'Google was officially founded in 1998. [article_id:1]'
                    
                    Context: Google, one of the largest tech companies in the United States, launched a new AI model in 2023.
                    Question:'q1_2 Which country is the company Google headquartered in?'
                    Answer: 'Google's headquarters is located in USA.[article_id:2]' 

                    Question:'q1_3 What is the stock ticker of Google or its listing status? Please provide only relevant details. If it's not a public listed company, please answer 'Google is not a public listed company.'
                    Answer: 'Google's parent company, Alphabet Inc., is listed on the stock market. The stock ticker for Alphabet is GOOGL & GOOG traded on the NASDAQ exchange in USA [article_id:3]'

                    Question:'q1_4 What type of business does the company Google provide?'
                    Answer: 'Google is a tech company that primarily focuses on online services, advertising, cloud computing, and hardware, while also venturing into various other sectors.[article_id:4]
        """
    )


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


class QuestionRelatedChunks(BaseModel):
    original_question: Optional[str]
    crime_id: Optional[int]
    time: Optional[str]
    subjects: Optional[List[str]]
    summary: Optional[str]
    adverse_info_type: Optional[List[str]]
    violated_laws: Optional[str]
    enforcement_action: Optional[str]


class StructuredDataChatReport(BaseModel):
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
