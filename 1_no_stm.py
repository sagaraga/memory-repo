
from langgraph.graph import StateGraph, START, MessagesState
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()    

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

graph = builder.compile()


result = graph.invoke({"messages": [{"role": "user", "content": "Hi! My name is Sagar."}]})

from pprint import pprint
pprint(result)

result2 = graph.invoke({"messages": [{"role": "user", "content": "what is my name?"}]})
pprint(result2)