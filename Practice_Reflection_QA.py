"""
Author: Rajendhiran Easu
Date: 11 May 2026
Description: Building Reflection Agent for QA & Improve with the feedback
"""

import re
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import StateGraph, START, END


def provide_llm(model: str, provider: str = "ollama", temperature: int = 0):
    return init_chat_model(model=model, model_provider=provider, temperature=temperature)


generate_llm = provide_llm(model="mistral:latest")
evaluate_llm = provide_llm(model="gpt-oss:20b")
improve_llm = provide_llm(model="llama3.2:latest")

GENERATE_DESCRIPTION = "generate_description"
EVALUATE_DESCRIPTION = "evaluate_description"
IMPROVE_DESCRIPTION = "improve_description"

GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        "You are an intelligent assistant to generate the content description for the Title: {title}, in 3 to 4 lines")
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


def is_article_to_be_improved(state: ArticleReflectionState) -> bool:
    """True → improve then re-evaluate. False → keep current description and end."""
    if state["score"] >= PASSING_EVAL_SCORE:
        print(" Score reached the level")
        return False
    if state["iteration_took"] > MAX_IMPROVE_ROUNDS_BEFORE_ACCEPT:
        print(" Max. iteration tried to refine")
        return False
    return True


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
        True: IMPROVE_DESCRIPTION,
        False: END
    })
    graph.add_edge(IMPROVE_DESCRIPTION, EVALUATE_DESCRIPTION)
    return graph


def run_agent():
    wf_app = define_workflow_graph().compile()
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

    # display(Image(wf_app.get_graph().draw_mermaid()))


run_agent()
