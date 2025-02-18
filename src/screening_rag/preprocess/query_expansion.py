from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List

# Define a pydantic model to enforce the output structure
class Questions(BaseModel):
    questions: List[str] = Field(
        description="A list of sub-questions related to the input query."
    )

# Create an instance of the model and enforce the output structure
model = ChatOpenAI(model="gpt-4o", temperature=0) 
structured_model = model.with_structured_output(Questions)

# Define the system prompt
system = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n"""

# Pass the question to the model
question = """ when did 陳水扁 commit a crime"""
questions = structured_model.invoke([SystemMessage(content=system)]+[HumanMessage(content=question)])
print(questions)