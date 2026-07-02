---
name: dev-blog-linkedin-post
description: >
  Generate a structured .md file with a blog post reference and LinkedIn post
  from a given code snippet, file, or technical concept. Use this skill whenever
  the user shares code, a file, or a concept and wants to create a blog reference
  document, a LinkedIn post, or both. Trigger even when the user says things like
  "write a post about this code", "create a blog reference for this", "generate
  LinkedIn content for this snippet", "document this for my blog", or "turn this
  into a post". Always use for any combination of code-to-content or
  file-to-content generation tasks.
---

# Dev Blog + LinkedIn Post Generator

Generates a clean, production-quality `.md` file from a code snippet, file, or
technical concept. Output has two sections: a blog post reference block and a
LinkedIn post block.

---

## Input

The user provides one or more of:
- A code snippet (any language)
- A file (uploaded or described)
- A technical concept, architecture, or approach
- Optional: preferred title, target audience, tone (defaults apply if omitted)

If the input is ambiguous (no code, no file, and no description), ask:
> "What's the code or concept you'd like to create content for?"

---

## Output Format

Generate a single `.md` file saved to `blogs_generated_md/<kebab-case-title>.md`
(create the folder if it doesn't exist), then call `present_files` with the path.

The file has this exact structure:

```
# <Title>

---

## 📝 Blog Post Reference

### Summary
<3–5 sentences: what the code/concept does, why it matters, and who it's for>

### Code Snippet
```<language>
<clean, well-commented snippet — extract or condense from the provided code>
```

### How It Works
<3–6 bullet points explaining the key mechanics, design choices, or architecture>
- Keep each bullet to 1–2 lines
- Focus on **what**, **why**, and **tradeoffs**
- Avoid restating what is already visible in the code

---

## 💼 LinkedIn Post

<Hook line — bold, punchy, under 12 words. No generic openers like "Excited to share".>

<2–3 sentences of context: what problem this solves or why it's relevant now.>

**Key takeaways:**
→ <Point 1 — specific and concrete>
→ <Point 2 — specific and concrete>
→ <Point 3 — specific and concrete>
→ <Point 4 — optional, only if genuinely distinct>

<One closing line — a call to action or reflection. Not a repeat of the hook.>

📖 Full blog post: [link]
💻 Code repo: [repo link]

<hashtags — 8–12, one line, space-separated, ordered: specific → broad>
```

---

## Generation Rules

### Blog Section
- **Title**: Derive from the code/concept. Use action framing: "How to X", "Building X", "X in Practice".
- **Summary**: Write for a mid-level developer audience unless the user specifies otherwise. Assume familiarity with the language but not the pattern.
- **Code Snippet**: 
  - If the provided code is long (>50 lines), extract the core logic only.
  - Add inline comments only where non-obvious.
  - Use the exact language of the provided code.
  - Wrap in a fenced code block with the correct language tag.
- **How It Works**:
  - Lead with the architectural insight, not the syntax.
  - If relevant, note what makes this approach different from the naive solution.
  - If there are meaningful tradeoffs (e.g., performance vs. readability), include one bullet for that.

### LinkedIn Section
- **Tone**: Direct, professional, confident. No hype. No filler.
- **Hook line**: Must create curiosity or surface a pain point. Must be bold (`**...**`).
- **Takeaway bullets**: Use `→` arrows. Keep to 1 line each. No nested bullets.
- **Closing line**: Distinct from the hook. Can be a question, a call to action, or an insight.
- **Link placeholders**: Always include `[link]` for blog and `[repo link]` for code repo — never omit.
- **Hashtags**: 8–12 tags. Start specific (e.g., `#LangGraph`, `#Kotlin`), end broad (e.g., `#SoftwareEngineering`, `#AI`). No duplicates. All lowercase with camelCase for multi-word tags.
- **No ending line repeats**: The closing line must not echo the hook, summary, or any bullet.

---

## Handling Special Cases

| Situation | Behaviour |
|---|---|
| Code is in Kotlin/Android | Tailor bullets to Jetpack Compose / Coroutines / MVVM context where relevant |
| Code is an AI/LangGraph workflow | Frame summary around agent architecture, not just code |
| No code provided, only a concept | Generate a conceptual snippet that illustrates the idea |
| User specifies audience (e.g., "for beginners") | Adjust summary depth and bullet verbosity accordingly |
| User provides a file | Read the file, extract the most illustrative portion, and proceed normally |

---

## Example Output Skeleton

```markdown
# Using LangGraph's Send API for Parallel Receipt Processing

---

## 📝 Blog Post Reference

### Summary
LangGraph's Send API enables true parallel fan-out within a graph by dynamically
dispatching work to a subgraph node for each item in a collection. Unlike a loop
inside a single node, each Send creates an independent branch that runs concurrently,
then fan-in via a reducer. This is ideal for batch processing tasks like receipt
scanning, where each item needs isolated LLM calls without blocking others.

### Code Snippet
```python
from langgraph.types import Send

def route_receipts(state: OverallState):
    # Fan-out: dispatch one Send per receipt
    return [
        Send("process_receipt", {"receipt": r})
        for r in state["receipts"]
    ]
```

### How It Works
- **Fan-out via Send**: Each `Send("node", payload)` spawns an independent branch
- **Isolated state**: Each branch receives only its own `receipt` slice, not the full state
- **Reducer fan-in**: Results collected into `overall_state["results"]` via `add` reducer
- **Concurrency**: All branches execute in parallel — no polling or manual thread management
- **Tradeoff**: Higher LLM call volume; offset by latency reduction at batch scale

---

## 💼 LinkedIn Post

**Parallel receipt processing in LangGraph — no threads, no queues.**

Most batch AI pipelines serialize work unnecessarily. LangGraph's Send API lets
you fan out to N parallel branches with one line — each branch isolated, each
running concurrently, all collected cleanly via a reducer.

**Key takeaways:**
→ Send API dispatches independent graph branches per item — true parallelism
→ State isolation prevents branch interference at the reducer boundary
→ Fan-in is declarative — just annotate your field with `Annotated[list, add]`
→ Replaces manual async orchestration for batch LLM workloads

Drop it into any batch pipeline and cut wall-clock time proportionally to batch size.

📖 Full blog post: [link]
💻 Code repo: [repo link]

#LangGraph #LangChain #AIEngineering #PythonDev #AgenticAI #MultiAgent #FastAPI #SoftwareEngineering #MLOps #AI
```

---

## Notes

- Always save to `blogs_generated_md/` (relative to project root) before calling `present_files`. Create the folder with `mkdir -p` if it doesn't exist.
- Filename: kebab-case version of the title, e.g., `using-langgraph-send-api.md`.
- If the user has a known LinkedIn formatting style (short, clean lines, bold text), honour it.
- Never pad the LinkedIn post to hit a word count — tighter is better.
