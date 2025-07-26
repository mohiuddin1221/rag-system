from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from utils.state import Agentstate
from utils.nodes import (
    question_rewriter,
    tools,
    generate_answer
    )


checkpointer = InMemorySaver()
workflow = StateGraph(Agentstate)
workflow.add_node("question_rewriter", question_rewriter)
workflow.add_node("embed_node", tools)
workflow.add_node("generate_answer", generate_answer)



###edge
workflow.add_edge(START, "question_rewriter")
workflow.add_edge("question_rewriter", "tools")
workflow.add_edge("tools", "generate_answer")
workflow.add_edge("generate_answer", END)




graph = workflow.compile(checkpointer=checkpointer)