from dataclasses import Field
from typing import List, TypedDict
from langgraph.messages import MessageState
from langchain_core.messages import BaseMessage
from openai import BaseModel


class Agentstate(TypedDict):
    question: str
    rephrased_question: str
    retrive_messages: List[BaseMessage]
    generate_answer: str

class answer(BaseModel):
    ansewr: str = Field(
        description = "answer always one word or teo word"
        
    )