"""
Author: Rajendhiran Easu
Date: 11 May 2026
Description: Building Reflection Agent for QA & Improve with the feedback
"""

import re
import io
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import StateGraph, START, END
from PIL import Image
from langgraph.graph.state import CompiledStateGraph


def provide_llm(model: str, provider: str = "ollama", temperature: float = 0.0):
    return init_chat_model(model=model, model_provider=provider, temperature=temperature)


generate_llm = provide_llm(model="mistral:latest")
evaluate_llm = provide_llm(model="gpt-oss:20b")
improve_llm = provide_llm(model="llama3.2:latest")

GENERATE_DESCRIPTION = "generate_description"
EVALUATE_DESCRIPTION = "evaluate_description"
IMPROVE_DESCRIPTION = "improve_description"

NEED_IMPROVEMENT = "need_improvement"
EVALUATION_PASS = "evaluation_pass"

GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        "You are an intelligent assistant to generate the content description for the Title: {title}, in 1 to 2 lines")
])

EVALUATOR_PROMPT = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        "You are an intelligent assistant to evaluate the content description generate with 1 to 2 lines and give SCORE (1 to 10) eg. format for score is (SCORE=7)  and improving feedback as separate.  Description: {description} for the Title: {title}")
])

IMPROVE_PROMPT = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        "You are an intelligent assistant to improve the content description generate with 1 to 2 lines with the provided feedback.  Description: {description} for the Title: {title}, feedback to improve: {feedback}")
])

# After evaluate: prefer score; only use iteration cap when score is still low.
PASSING_EVAL_SCORE = 9  # score >= this → use result, done
# When score < PASSING_EVAL_SCORE: improve while iteration_took <= this; above → use latest result anyway
MAX_IMPROVE_ROUNDS_BEFORE_ACCEPT = 2


class ArticleReflectionState(TypedDict):
    title: str
    description: str
    feedback: str
    score: int
    iteration_took: int


def _parse_evaluator_score(text: str, default: int = 5) -> int:
    """Extract SCORE (1–10) from evaluator LLM output; clamp to valid range."""
    if not text:
        return default
    match = re.search(r"(?i)SCORE\s*[=:]\s*(\d+)", text)
    if not match:
        return default
    try:
        score = int(match.group(1))
    except ValueError:
        return default
    return max(1, min(10, score))


def is_article_to_be_improved(state: ArticleReflectionState) -> str:
    """False when good enough: if either exit condition passes, keep the current result."""
    score_passed = state["score"] >= PASSING_EVAL_SCORE
    iteration_cap_reached = state["iteration_took"] >= MAX_IMPROVE_ROUNDS_BEFORE_ACCEPT
    if score_passed or iteration_cap_reached:
        return EVALUATION_PASS
    return NEED_IMPROVEMENT


def generate_article_description(state: ArticleReflectionState):
    print("\n Request to Generate")
    article_title = state["title"]
    chain = GENERATOR_PROMPT | generate_llm
    g_res = chain.invoke({"title": article_title})
    state["description"] = g_res.content
    print(" Description Generated")
    return state


def evaluate_article_description(state: ArticleReflectionState):
    print("\n Request to Evaluate")
    article_title = state["title"]
    article_desc = state["description"]
    chain = EVALUATOR_PROMPT | evaluate_llm
    eval_result = chain.invoke({"title": article_title, "description": article_desc}).content
    state["feedback"] = eval_result
    score = _parse_evaluator_score(eval_result)
    state["score"] = score
    print(f" Parsed score: {score}")
    return state


def improve_article_description(state: ArticleReflectionState):
    print("\n Request to Improve")
    article_title = state["title"]
    article_desc = state["description"]
    article_feedback = state["feedback"]
    chain = IMPROVE_PROMPT | improve_llm
    imp_result = chain.invoke(
        {"title": article_title, "description": article_desc, "feedback": article_feedback}).content
    state["description"] = imp_result
    state["iteration_took"] += 1
    print(" Description Improvised")
    return state


def define_workflow_graph():
    graph = StateGraph(ArticleReflectionState)
    graph.add_node(GENERATE_DESCRIPTION, generate_article_description)
    graph.add_node(EVALUATE_DESCRIPTION, evaluate_article_description)
    graph.add_node(IMPROVE_DESCRIPTION, improve_article_description)

    graph.add_edge(START, GENERATE_DESCRIPTION)
    graph.add_edge(GENERATE_DESCRIPTION, EVALUATE_DESCRIPTION)
    graph.add_conditional_edges(EVALUATE_DESCRIPTION, is_article_to_be_improved, {
        NEED_IMPROVEMENT: IMPROVE_DESCRIPTION,
        EVALUATION_PASS: END
    })
    graph.add_edge(IMPROVE_DESCRIPTION, EVALUATE_DESCRIPTION)
    return graph

def render_store_graph(app:CompiledStateGraph):
    try:
        graph_bytes = app.get_graph().draw_mermaid_png()

        # Process the bytes using Pillow
        image = Image.open(io.BytesIO(graph_bytes))

        # OPTION A: Save it directly into your PyCharm project directory
        # It will instantly appear in your left-hand project sidebar as a PNG!
        image.save("graph/reflection_graph.png")
        print("Success! Double-click 'reflection_graph.png' in the PyCharm sidebar to view.")

        # OPTION B: Uncomment the line below to pop it open in the macOS Preview App
        # image.show()

    except Exception as e:
        print(f"Failed to render image. Error: {e}")


def run_agent():
    wf_app = define_workflow_graph().compile()
    #render_store_graph(wf_app)
    while True:
        title_input = str(input("\n Article Title Please: "))
        if title_input == "/bye":
            print(" Thank you, bye!")
            break
        initial_state = {
            "title": f"{title_input}",
            "description": "",
            "feedback": "",
            "score": 0,
            "iteration_took": 0
        }
        result = wf_app.invoke(initial_state)
        print(f"\n Result: {result}")


if __name__ == "__main__":
    run_agent()
