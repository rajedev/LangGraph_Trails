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

exception_action = True


# manual conversation w/o checkpointer and config - checks
# convo = []

class MsgState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_here(state: MsgState):
    u_msg = state['messages']
    log = u_msg[-1].content.split("-")
    # print(f"log={log}")
    u_msg = log[-1]
    global exception_action
    if exception_action == True and "exception" in log:
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
    global exception_action
    while True:
        usr_input = str(input("User: "))
        us = usr_input.split("-")
        # print(us)
        if "exception " in us:
            exception_action = True
        if usr_input == "er":
            exception_action = False
        if usr_input == "thread":
            # conversation = checkpointer.list(None)
            conversation = checkpointer.list(config=config)
            for co in conversation:
                # print(co[1]['channel_values'])
                print(co[0])
            continue
        if usr_input == "snap":
            snap = list(graph.get_state_history(config=config))
            for s in snap:
                print(s.values)
            continue
        if usr_input in ["/bye", "exit", "quit"]:
            break
        convo_input = None if usr_input == "er" else {
            "messages": [HumanMessage(content=usr_input)]
            # "messages": convo + [HumanMessage(content=usr_input)]
        }
        try:
            result = graph.invoke(input=convo_input, config=config)
            # result = graph.invoke(input=initial_state)
            print(f"AI: {result["messages"][-1].content}")
        except Exception as e:
            print(f"\nException: {e}\n")


if __name__ == "__main__":
    execute_graph()

# Conversation for references - Exception raised & AI response from the last leaved convo on "er" to clear the error
"""

User: hi
AI: Hello! How can I assist you today? 😊
User: exception-this is rajesh, works as developer, suggest me with 20 words on carrer advice

Exception: Err Checker -- Manual Invoke

User: thread
....
User: snap
....
User: er
AI: Continuously learn new languages, contribute to open source, master algorithms, network actively, prioritize code quality, seek mentorship, and embrace feedback.
User: /bye

"""
