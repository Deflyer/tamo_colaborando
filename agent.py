import os
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages : List[HumanMessage | AIMessage]

llm = ChatOpenAI(
    model="nvidia/nemotron-nano-12b-v2-vl:free",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def process_message(state: AgentState) -> AgentState:
    """Invoke the llm model appending the messages into
    a context list"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"Current state: {state['messages']}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process_message)
graph.add_edge(START, "process")
graph.add_edge("process", END)

chat = graph.compile()

user_input = str(input('Prompt: '))
conversational_history = []
while user_input != 'exit':
    conversational_history.append(user_input)
    result = chat.invoke({"messages": conversational_history})
    print(result["messages"])

    user_input = str(input('Prompt: '))



