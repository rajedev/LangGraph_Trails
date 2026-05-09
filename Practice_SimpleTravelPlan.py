"""
Author: Rajendhiran Easu
Date: 09 May 2026
Description: 
"""

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

flights = [{
    "flight": "indigo",
    "source": "Chennai",
    "destination": "Bangalore",
    "ticket_available": 3
}, {
    "flight": "spicejet",
    "source": "Chennai",
    "destination": "Bangalore",
    "ticket_available": 0
}, {
    "flight": "air india",
    "source": "Chennai",
    "destination": "Bangalore",
    "ticket_available": 2
}]

llm = init_chat_model(model="gpt-oss:20b", model_provider="ollama", temperature=0)

sys_prompt = SystemMessage(content=(
    "You are an intelligent travel assistant. Answer only using tool results; do not invent flights or counts.\n"
    "Tool choice:\n"
    "- If the user names a specific airline or flight (e.g. Indigo, SpiceJet, Air India), call check_availability "
    "with that name. Do this even if they also mention a city or route.\n"
    "- If they only ask for a general list of flights or carriers with seats to a destination (no specific airline named), "
    "call get_list_of_flights with the destination city.\n"
    "If one tool already returned enough to answer, you may reply without calling more tools."
))

TEST_PROMPT1 = "Provide me a list of flights available to travel Bangalore?"
TEST_PROMPT2 = "How many tickets are available to travel Bangalore with Indigo?"
TEST_PROMPT3 = "I would like to travel to Bangalore in spicejet, tickets are available?"

class MsgState(MessagesState):
    pass


def _norm(s: str) -> str:
    """Strip whitespace and casefold for stable, case-insensitive comparisons."""
    return (s or "").strip().casefold()


def check_availability(flight_name: str) -> str:
    """Check ticket availability for one named airline or flight operator.
"STRICTLY respond to the tools result, don't infer anything extra about the availability or suggesting other travel plan or flights"
    Use this when the user mentions a specific carrier by name (e.g. indigo, spicejet, air india),
    including questions like 'is X available?' or 'tickets on X?'. Not for open-ended 'list all flights'."""

    query = _norm(flight_name)
    if not query:
        return "Flight is unavailable"
    for data in flights:
        if query in _norm(data["flight"]):
            if data["ticket_available"] > 0:
                return f"{data['flight']} is available with {data['ticket_available']} tickets"
    return "Flight is unavailable"


def get_list_of_flights(destination: str) -> list:
    """List carriers that have at least one ticket to the given destination.
    "STRICTLY respond to the tools result, don't infer anything extra about the availability or suggesting other travel plan or flights"
    Use only when the user wants a general list of options to a city and does not single out one airline by name."""
    query = _norm(destination)
    if not query:
        return []
    list_of_flights = []
    for data in flights:
        if query in _norm(data["destination"]) and data["ticket_available"] > 0:
            list_of_flights.append(data["flight"])
    return list_of_flights


# check_availability first: models often skim tool order; specific carrier questions should prefer it.
_travel_tools = [check_availability, get_list_of_flights]
llm_with_tools = llm.bind_tools(_travel_tools)


def tool_call_execution(message_state: MsgState):
    #print(f"state msg: {message_state['messages']}")
    return {"messages": [llm_with_tools.invoke([sys_prompt] + message_state["messages"])]}


def build_graph() -> CompiledStateGraph:
    workflow = StateGraph(MsgState)
    workflow.add_node("execution", tool_call_execution)
    workflow.add_node("tools", ToolNode(_travel_tools))

    workflow.add_edge(START, "execution")
    workflow.add_conditional_edges("execution", tools_condition)
    workflow.add_edge("tools", "execution")

    graph = workflow.compile()

    return graph

def run_agent():
    graph = build_graph()
    # result = graph.invoke({"messages":[HumanMessage(content=TEST_PROMPT3)]})
    # for msg in result["messages"]:
    #     msg.pretty_print()
    while True:
        user_convo = input("User: ")
        if user_convo.casefold() == "/bye":
            print("Thank you, bye!")
            break
        res = graph.invoke({"messages":[HumanMessage(content=user_convo)]})
        #print(res["messages"][-1].content)
        for msg in res["messages"]:
            msg.pretty_print()

run_agent()