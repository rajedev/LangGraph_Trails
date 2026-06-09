"""
Author: Rajendhiran Easu
Date: 04 Mar 2026
Description:
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

INIT_EXECUTION = "init_execute"
CHECK_TOOL = "tools"

sys_prompt = SystemMessage(content=(
    "You are an HR assistant. You MUST answer only from tool results.\n"
    "STRICT rules:\n"
    "- Never invent employee names, IDs, salaries, or departments.\n"
    "- If no tool has been called yet for the question, call the appropriate tool first.\n"
    "- If tool results are empty ([]), say no employees match — do not guess.\n"
    "- Do not pass made-up data into tool arguments; only pass values from the user's question.\n"
    "- Reply using only fields returned by tools: id, name, sal, dept.\n"
    "Salary amounts are in INR. 1 lakh = 100,000 (e.g. 30 lakhs = 3,000,000).\n"
    "Tool choice:\n"
    "- Salary/pay/earnings criteria → filter_sal(amount).\n"
    "- Department/team criteria → filter_dep(department).\n"
    "When the user gives BOTH salary and department criteria, call every relevant tool before answering.\n"
    "Combine tool results:\n"
    "- 'above X salary AND in dept Y' → employees in both results.\n"
    "- 'above X salary AND NOT in dept Y' → salary results excluding department results.\n"
    "Only give the final answer after all required tools have returned."
))

employees = [{
    "id": "E124",
    "name": "Rajiv",
    "sal": 1500000.0,
    "dept": "HR"
}, {
    "id": "E112",
    "name": "Ganesh",
    "sal": 2500000.0,
    "dept": "Technician"
}, {
    "id": "E124",
    "name": "Muru",
    "sal": 3500000.0,
    "dept": "Finance"
}
]

load_dotenv(override=True)
# llm = init_chat_model(
#     model="nvidia/nemotron-3-nano-30b-a3b",
#     model_provider="nvidia",
#     temperature=0.0)

llm = init_chat_model(model="gpt-oss:20b", model_provider="ollama", temperature=0)


class UserMsg(MessagesState):
    pass


def initialize(state: UserMsg):
    c_msg = llm_with_tools.invoke([sys_prompt] + state["messages"])
    return {"messages": [c_msg]}


def search_employee_by_salary(amount: float) -> list[dict]:
    """Return employees whose salary is strictly greater than the given amount (INR).
    Use for pay/earnings filters. Can be combined with search_employee_by_dep when the
    question also mentions a department (e.g. 'above 20 lakhs and not in Finance')."""
    return [e for e in employees if e["sal"] > amount]


def search_employee_by_dep(department: str) -> list[dict]:
    """Return employees in the given department (case-insensitive match on dept).
    Use for team/department filters. Can be combined with search_employee_by_salary when
    the question also mentions salary (e.g. to include or exclude a department)."""
    return [e for e in employees if department.lower() == e["dept"].lower()]


_search_tools = [search_employee_by_dep, search_employee_by_salary]
llm_with_tools = llm.bind_tools(_search_tools)


# list_of_available_tools = {
#     "filter_dep": search_employee_by_dep,
#     "filter_sal": search_employee_by_salary,
# }
#
# # LLM tool names must match the dict keys above
# _search_tools = [
#     StructuredTool.from_function(func=fn, name=name, description=fn.__doc__ or "")
#     for name, fn in list_of_available_tools.items()
# ]
#
# llm_with_tools = llm.bind_tools(_search_tools)
#
#
# def trigger_tool_call(state: UserMsg):
#     tool_messages = []
#
#     for tool_call in state["messages"][-1].tool_calls:
#         name = tool_call["name"]
#         fn = list_of_available_tools.get(name)
#         if fn is None:
#             content = f"Error: unknown tool '{name}'"
#         else:
#             try:
#                 result = fn(**tool_call["args"])
#                 content = result if isinstance(result, str) else json.dumps(result)
#             except TypeError as exc:
#                 content = f"Error: invalid arguments for '{name}': {exc}"
#
#         tool_messages.append(
#             ToolMessage(content=content, name=name, tool_call_id=tool_call["id"])
#         )
#
#     return {"messages": tool_messages}
#
#
# def if_tool_call_required(state: UserMsg):
#     last_msg = state["messages"][-1]
#
#     if last_msg.tool_calls:
#         return "tools"
#     else:
#         return "end"


def build_workflow():
    wf_graph = StateGraph(UserMsg)
    wf_graph.add_node(INIT_EXECUTION, initialize)
    wf_graph.add_node(CHECK_TOOL, ToolNode(_search_tools))
    # wf_graph.add_node(CHECK_TOOL, trigger_tool_call)

    wf_graph.add_edge(START, INIT_EXECUTION)
    wf_graph.add_conditional_edges(INIT_EXECUTION, tools_condition)
    wf_graph.add_edge(CHECK_TOOL, INIT_EXECUTION)
    # wf_graph.add_conditional_edges(INIT_EXECUTION, if_tool_call_required, {
    #     "tools": CHECK_TOOL,
    #     "end": END
    # })
    return wf_graph


if __name__ == "__main__":
    graph = build_workflow().compile()
    while True:
        us_input = input("User: ('/bye' to exit): ")
        if us_input == "/bye":
            break
        result = graph.invoke({"messages": [HumanMessage(content=us_input)]})
        for msg in result["messages"]:
            msg.pretty_print()
