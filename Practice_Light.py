"""
Author: Rajendhiran Easu
Date: 12 July 2026
Description: 
"""
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()
llm = init_chat_model(model="nvidia/nemotron-3-nano-30b-a3b", model_provider="nvidia", temperature=0.0)
config = {
    "configurable": {
        "thread_id": "usr_1"
    }
}
llm_with_config = llm.with_config(configurable={
    "max_tokens": 5
})
checkpointer = InMemorySaver()

# manual conversation w/o checkpointer and config - checks
# convo = []

class MsgState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_here(state: MsgState):
    u_msg = state['messages']
    if u_msg[-1].content == "exception":
        raise Exception("Err Checker -- Manual Invoke")
    # convo.extend(u_msg)
    # print(u_msg)
    res = llm.invoke(u_msg)
    # print(res)
    return {
        'messages': [res]
    }


def workflow_graph():
    wf_graph = StateGraph(MsgState)
    wf_graph.add_node("init_chat", chat_here)
    wf_graph.add_edge(START, "init_chat")
    wf_graph.add_edge("init_chat", END)

    # return wf_graph.compile()
    return wf_graph.compile(checkpointer=checkpointer)


def execute_graph():
    graph = workflow_graph()
    while True:
        usr_input = str(input("User: "))
        if usr_input == "thread":
            #conversation = checkpointer.list(None)
            conversation = checkpointer.list(config=config)
            for co in conversation:
                #print(co[1]['channel_values'])
                print(co[0])
            continue
        if usr_input == "snap":
            snap = graph.get_state(config=config)
            print(snap)
            continue
        if usr_input in ["/bye", "exit", "quit"]:
            break
        initial_state = {
            "messages": [HumanMessage(content=usr_input)]
            # "messages": convo + [HumanMessage(content=usr_input)]
        }
        try:
            result = graph.invoke(input=initial_state, config=config)
            # result = graph.invoke(input=initial_state)
            print(f"AI: {result["messages"][-1].content}")
        except Exception as e:
            print(f"\n\n Except: {e}")


if __name__ == "__main__":
    execute_graph()
