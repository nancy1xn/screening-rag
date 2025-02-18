from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List

# Define a pydantic model to enforce the output structure
class Is_adverse_media(BaseModel):
    is_adverse_media: bool = Field(
        description="Check if the contents contains adverse media related to financial crime. If so, please add a tag is-negative-news: True, otherwise, add a tag is-negative-news: False. The definition of financial crime related to adverse media is as following (Media coverage trigger: Regulatory Enforcement Action/ Convictions/ Ongoing investigations or allegations about a financial crime related category): money laundering/bribery and corruption/fraud or weakness in the client's fraud prevention controls/ stock exchange irregularities, insider trading, or market manipulation/ accounting irregularities/ tax evasion and other tax crimes(related to direct taxes and indirect taxes/regulatory enforcement action against an entity in the regulated sector that may have links to the client/major sanctions breaches or violations, including dealings with or other involvement with sanctioned countries or territories or countries subject to extensive sanction regimes/terrorism, including terrorist financing/illicit trafficking in narcotic drugs and psychotropic substances, arms trafficking or stolen goods/ smuggling(including in relation to customs and excise duties and taxes)/ trafficking in human beings or migrant smuggling/sexual exploitation of child labor/ extortion/counterfeiting currency, forgery, or piracy of products/organized crime or racketeering/ benefitting from serious offences including kidnapping, illegal restraint and hostage-taking, robbery, theft, murder or causing grievous bodily injury/ Benefitting from environmental crime/ Benefitting from other unethical or criminal behaviour))"
    )

# Create an instance of the model and enforce the output structure
model = ChatOpenAI(model="gpt-4o", temperature=0) 
structured_model = model.with_structured_output(Is_adverse_media)

# Define the system prompt
system = """You are a helpful assistant to check if the contents contains adverse media related to financial crime. If so, please add a tag is-negative-news: yes. You do not need to answer other things except of the tag"""

# Pass the question to the model
news = """Mr Mao LIN SHIH was indicted for embezzlement by the special investigation  division of the supreme prosecutors office on july 15, 2008. He was alleged to have embezzled by using false reciepts to claim expenses ranged from about $12000 to $200000"""
is_adverse_media= structured_model.invoke([SystemMessage(content=system)]+[HumanMessage(content=news)])
print(is_adverse_media)