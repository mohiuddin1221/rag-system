from typing import Annotated, List
from state import Agentstate, answer
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from data.embed import  embeddings, index
from langgraph.prebuilt import ToolNode, InjectedState


llm = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_type="azure",
    temperature=1,
)




def question_rewriter(state:Agentstate ):
    print(f"Question rewriter agent running")
    current_question = state["question"]
    messages = [
        SystemMessage(
            content = "You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval."
        ),
        HumanMessage(
            content = current_question
        )

    ]

    repharse_format = ChatPromptTemplate.from_messages(messages)
    response = repharse_format | llm
    better_question = response.content()

    print(f"Rephased Qusetion: {better_question}")
    state["rephrased_question"] = better_question

    return state


@tool
def get_answer_from_rag(input: str,state: Annotated[dict, InjectedState]) -> str:
    """
    Retrieve relevant answers from RAG vector store based on input query.

    Args:
        input (str): The input query string to search in the vector store.
        namespace (str, optional): The Pinecone namespace to query. Defaults to "tenminute-school".

    Returns:
        List[Any]: A list of similar answer results with metadata.
    """
    query_embedding = embeddings.embed_query(input)
    data = index.query(
        vector=query_embedding,
        top_k=4,
        include_metadata=True,
        
    )
    state["retrive_messages"] = data
    return state




tools = [get_answer_from_rag]
tools = ToolNode(tools)





def generate_answer(state: Agentstate):
    rephrased_question = state["rephrased_question"]
    retrive_messages =  state["retrive_messages"]

    context = "\n".join(
        [doc["metadata"]["text"] for doc in retrive_messages["matches"] if "text" in doc["metadata"]]
    )
    messages = [
    SystemMessage(
"""
You are a smart assistant designed to answer user questions concisely.

Instructions:
- You will be given a rephrased question and some retrieved context messages from a knowledge source.
- Your job is to generate a short, precise answer in one line.
- The answer must directly address the question using the retrieved messages.

Example format:
rephrased_question: What class do you read in?
answer: Class five.

Be accurate, brief, and relevant. Do not provide explanations. Do not repeat the question.
"""
    ),
    HumanMessage(
         content=f"rephrased_question: {rephrased_question}\n\ncontext:\n{context}"
        
    )
    ]

    grade_format = ChatPromptTemplate.from_messages(messages)
    structured_llm = llm.with_structured_output(answer)
    response = grade_format |  structured_llm
    result = response.invoke({})

    state["generate_answer"] = result.content()
    return state







