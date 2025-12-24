# Self-Refining AI using LangChain Runnables

This project demonstrates how to build a **self-refining LLM pipeline** using **LangChain Core Runnables**.
The system generates an answer, evaluates it across multiple dimensions, synthesizes feedback, and then improves the original response based on that feedback.

The focus is on **Runnable composition**, not agents, tools, or memory.

---

## 🔍 What This Project Does

For a given user question, the system:

1. **Generates an initial answer**
2. **Evaluates the answer in parallel** on:

   * Quality
   * Clarity
   * Completeness
3. **Combines all evaluations into a single, actionable feedback**
4. **Refines the original answer** using the combined feedback

This mimics a real-world editorial workflow: *write → review → revise*.

---

## 🧠 Key Concepts Used

* `RunnableSequence` (via pipe `|`)
* `RunnableParallel` for multi-dimension evaluation
* Prompt chaining and output parsing
* Feedback loops in LLM systems

No agents, no tools — **pure LangChain Core**.

---

## 🏗️ High-Level Flow

```
User Question
   ↓
Initial Answer
   ↓
Parallel Review (Quality | Clarity | Completeness)
   ↓
Combined Feedback
   ↓
Improved Answer
```


## ▶️ How to Run

1. Set up environment variables (`.env`) for Google GenAI
2. Install dependencies
3. Run the script and enter a question when prompted

---

## 📌 Notes

* The system always refines once (no stopping condition by design)
* Designed for learning LangChain internals, not production use

---

## 📖 Takeaway

This project shows how **analysis → feedback → action** can be composed cleanly using LangChain Runnables, forming the foundation for more advanced LLM systems.

