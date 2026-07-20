"""
Author: Rajendhiran Easu
Date: 10 June 2026
Description: Command-based routing demo — classify user intent (call / teams / email)
             with structured output, then route via langgraph.types.Command.goto.
"""
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv(override=True)

llm = init_chat_model(model="nvidia/nemotron-3-nano-30b-a3b", model_provider="nvidia", temperature=0.0)

INTENT_SYSTEM = (
    "You extract how the user wants support to reach them AND their contact value.\n"
    "Return:\n"
    "1) action_intent — exactly one of:\n"
    "   - call: wants a phone call / ring / dial\n"
    "   - teams: wants Microsoft Teams / Teams chat\n"
    "   - email: wants email / mail\n"
    "   - ignore: greeting only, unclear request, or no contact mode\n"
    "2) contact_detail — the contact value matching that mode:\n"
    "   - call → phone number (e.g. +91-9876543210)\n"
    "   - teams → Teams id / handle\n"
    "   - email → email address\n"
    "   - ignore → empty string \"\"\n"
    "If the mode is clear but no contact value is given, still set action_intent "
    "and leave contact_detail as \"\".\n"
    "Do not invent a contact value that the user did not provide."
)


class UserIntent(BaseModel):
    action_intent: Literal["call", "teams", "email", "ignore"] = Field(
        default="ignore",
        description=(
            "Contact mode: 'call' (phone), 'teams' (Microsoft Teams), "
            "'email', or 'ignore' if unclear / no mode."
        ),
    )
    contact_detail: str = Field(
        default="",
        description=(
            "Contact value from the user message for the chosen mode: "
            "phone number for call, Teams id for teams, email address for email. "
            "Empty string if not provided or action_intent is ignore."
        ),
    )


llm_with_intent = llm.with_structured_output(UserIntent)


class ReachSupportState(MessagesState):
    contact_number: str
    teams_id: str
    email_id: str
    status: str


def trigger_action(state: ReachSupportState):
    # print(usr_msg)
    intent: UserIntent = llm_with_intent.invoke([
        SystemMessage(content=INTENT_SYSTEM),
        state["messages"][-1],
    ])
    # print(intent)
    update_next_value = {}
    update_next = "ignore"
    match intent.action_intent:
        case "call":
            update_next_value = {
                "contact_number": intent.contact_detail
            }
            update_next = "call"
        case "teams":
            update_next_value = {
                "teams_id": intent.contact_detail
            }
            update_next = "teams_call"
        case "email":
            update_next_value = {
                "email_id": intent.contact_detail
            }
            update_next = "email"
        case _:
            update_next = "ignore"

    return Command(update=update_next_value, goto=update_next)


def make_teams_call(state: ReachSupportState):
    return {
        "status": f"You will get support via Teams to the id: {state['teams_id']}"
    }


def make_a_call(state: ReachSupportState):
    return {
        "status": f"We will ring back you shortly to {state['contact_number']}"
    }


def send_email(state: ReachSupportState):
    return {
        "status": f"We will reach out to your email id: {state['email_id']}"
    }


def not_reachable(_: ReachSupportState):
    return {
        "status": "Welcome & Thanks, share your mode of reach for the query"
    }


def workflow_builder() -> CompiledStateGraph:
    workflow = StateGraph(ReachSupportState)
    workflow.add_node("trigger", trigger_action)
    workflow.add_node("call", make_a_call)
    workflow.add_node("teams_call", make_teams_call)
    workflow.add_node("email", send_email)
    workflow.add_node("ignore", not_reachable)

    workflow.add_edge(START, "trigger")

    return workflow.compile()


def execution():
    graph = workflow_builder()
    while True:
        usr_input_msg = str(input("User - (/bye to exit): "))
        if usr_input_msg == "/bye":
            break
        result = graph.invoke({
            "messages": [HumanMessage(content=usr_input_msg)]
        })
        # print(result)
        print(f"AI: {result['status']}")


if __name__ == "__main__":
    execution()
