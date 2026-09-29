"""
Author: Rajendhiran Easu
Date: 24 July 2026
Description: 
"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# 1. State Definition
class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]
    task: str


# Worker subgraphs must NOT inherit parent `messages`.
# If they do, returning the full list + parent `operator.add` duplicates history.
class WorkerInput(TypedDict):
    task: str


class WorkerOutput(TypedDict):
    messages: Annotated[list[str], operator.add]


class WorkerState(TypedDict):
    task: str
    messages: Annotated[list[str], operator.add]


# 2. Worker Subgraphs (The Specialized Agents)
#    Worker 1: Coder
def coder_node(state: WorkerState):
    print("--- CODER WORKING ---")
    return {"messages": [f"Coder: Written clean Python code for the task: {state['task']}."]}


coder_builder = StateGraph(WorkerState, input_schema=WorkerInput, output_schema=WorkerOutput)
coder_builder.add_node("write_code", coder_node)
coder_builder.add_edge(START, "write_code")
coder_subgraph = coder_builder.compile()


# Worker 2: Tester
def tester_node(state: WorkerState):
    print("--- TESTER WORKING ---")
    return {"messages": [f"Tester: All unit tests passed successfully for the task: {state['task']}."]}


tester_builder = StateGraph(WorkerState, input_schema=WorkerInput, output_schema=WorkerOutput)
tester_builder.add_node("run_tests", tester_node)
tester_builder.add_edge(START, "run_tests")
tester_subgraph = tester_builder.compile()


# ==========================================
# 3. The Supervisor (The Parent Orchestrator)
# ==========================================
def supervisor_node(state: AgentState) -> Command[Literal["coder", "tester", "__end__"]]:
    print("--- SUPERVISOR ORCHESTRATING ---")

    # Simple rule-based logic (Replace this with an LLM call in production)
    history = "".join(state.get("messages", []))

    if "Coder:" not in history:
        print("Supervisor: Routing to Coder.")
        return Command(goto="coder")

    elif "Tester:" not in history:
        print("Supervisor: Routing to Tester.")
        return Command(goto="tester")

    else:
        print("Supervisor: Work complete. Ending graph.")
        return Command(goto=END)


# ==========================================
# 4. Building the Main Graph
# ==========================================
workflow = StateGraph(AgentState)

# Add the supervisor node
workflow.add_node("supervisor", supervisor_node)

# Add the subgraphs as nodes
workflow.add_node("coder", coder_subgraph)
workflow.add_node("tester", tester_subgraph)

# Setup entry point + return to supervisor after each worker
workflow.add_edge(START, "supervisor")
workflow.add_edge("coder", "supervisor")
workflow.add_edge("tester", "supervisor")

# Compile the graph
app = workflow.compile()

if __name__ == "__main__":
    # Run the graph
    initial_state = {"task": "Build a registration API", "messages": []}
    final_state = app.invoke(initial_state)

    print("\n--- FINAL CHAT HISTORY ---")
    for msg in final_state["messages"]:
        print(msg)
