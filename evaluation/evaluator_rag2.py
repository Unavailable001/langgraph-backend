"""Evaluate RAG answers with Gemini acting as an LLM judge.

Run from the project root with:
    python -m evaluation.evaluator_rag2

Requires GOOGLE_API_KEY (or GOOGLE_GENERATIVE_AI_API_KEY) for the judge model.
"""

import json
import os
import re
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from appRag.main import run_rag

load_dotenv()

JUDGE_MODEL = "gemini-3.5-flash-lite"
SCORE_THRESHOLD = 4
ABSTENTION = "I don't know based on the provided documents."

# `answerable` is the expected retrieval label for each evaluation question.
EVALUATION_CASES = [
    {
        "question": "What is Java?",
        "reference_answer": (
            "Java is a high-level, object-oriented programming language."
        ),
        "answerable": True,
    },
    {
        "question": "What are the main features of Java?",
        "reference_answer": (
            "Java provides object-oriented programming, platform independence, "
            "security, robustness, and automatic memory management."
        ),
        "answerable": True,
    },
    {
        "question": "What is the capital of France?",
        "reference_answer": ABSTENTION,
        "answerable": False,
    },
]

JUDGE_PROMPT = """You are an impartial evaluator for a retrieval-augmented question-answering system.
Evaluate the candidate answer using the question, retrieved context, and reference answer.
Do not use outside knowledge. Treat the reference answer as the expected answer when
one is provided, while also checking that claims are supported by the context.

Return ONLY one JSON object with this exact schema:
{{
  "answerable": true,
  "correctness": 1,
  "relevance": 1,
    "retrieval_relevance": 1,
  "groundedness": 1,
  "reasoning": "brief explanation"
}}

Rules:
- answerable: true only if the retrieved context contains enough information to answer the question.
- correctness: integer 1-5; factual accuracy and agreement with the reference answer.
- relevance: integer 1-5; how directly the candidate answers the question.
- retrieval_relevance: integer 1-5; how relevant the retrieved context is to the question
    and whether it contains useful information for answering it.
- groundedness: integer 1-5; how well candidate claims are supported by retrieved context.
- An appropriate abstention when context is insufficient is relevant and grounded, but the
  answerable field must still be false.
- Do not award high scores for a plausible answer that is unsupported by the supplied context.

Question:
{question}

Reference answer:
{reference_answer}

Retrieved context:
{context}

Candidate answer:
{candidate_answer}
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object returned directly or inside a Markdown code fence."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise ValueError("The judge response did not contain a JSON object.")
        cleaned = cleaned[start : end + 1]

    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("The judge response must be a JSON object.")
    return parsed


def _validated_judgment(raw: dict[str, Any]) -> dict[str, Any]:
    answerable = raw.get("answerable")
    if not isinstance(answerable, bool):
        raise ValueError("Judge field 'answerable' must be a boolean.")

    judgment: dict[str, Any] = {"answerable": answerable}
    for score_name in (
        "correctness",
        "relevance",
        "retrieval_relevance",
        "groundedness",
    ):
        score = raw.get(score_name)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"Judge field '{score_name}' must be a number from 1 to 5.")
        if not 1 <= score <= 5:
            raise ValueError(f"Judge field '{score_name}' must be between 1 and 5.")
        judgment[score_name] = float(score)

    reasoning = raw.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise ValueError("Judge field 'reasoning' must be a string.")
    judgment["reasoning"] = reasoning
    return judgment


def grade_answer(
    grader: ChatGoogleGenerativeAI,
    *,
    question: str,
    reference_answer: str,
    context: str,
    candidate_answer: str,
) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        context=context or "[No context was retrieved.]",
        candidate_answer=candidate_answer,
    )
    response = grader.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )
    if not isinstance(content, str):
        raise ValueError("The judge returned an unsupported response format.")
    return _validated_judgment(_extract_json_object(content))


def _binary_metrics(actual: list[int], predicted: list[int]) -> dict[str, Any]:
    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    tn = sum(a == 0 and p == 0 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_rag_with_judge() -> dict[str, Any]:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GOOGLE_API_KEY (or GOOGLE_GENERATIVE_AI_API_KEY) to use the Gemini judge."
        )

    grader = ChatGoogleGenerativeAI(
        model=JUDGE_MODEL,
        temperature=0,
        google_api_key=api_key,
    )
    results = []
    actual_answerable: list[int] = []
    predicted_answerable: list[int] = []

    for case in EVALUATION_CASES:
        rag_result = run_rag(case["question"])
        candidate_answer = str(rag_result.get("answer", ""))
        context = str(rag_result.get("context", ""))
        judgment = grade_answer(
            grader,
            question=case["question"],
            reference_answer=case["reference_answer"],
            context=context,
            candidate_answer=candidate_answer,
        )
        actual_label = int(case["answerable"])
        predicted_label = int(judgment["answerable"])
        actual_answerable.append(actual_label)
        predicted_answerable.append(predicted_label)

        result = {
            "question": case["question"],
            "reference_answer": case["reference_answer"],
            "candidate_answer": candidate_answer,
            "expected_answerable": case["answerable"],
            "judge": judgment,
            "answerable_prediction_correct": actual_label == predicted_label,
            "answer_quality_pass": all(
                judgment[name] >= SCORE_THRESHOLD
                for name in (
                    "correctness",
                    "relevance",
                    "retrieval_relevance",
                    "groundedness",
                )
            ),
        }
        results.append(result)

    metrics = _binary_metrics(actual_answerable, predicted_answerable)
    return {
        "judge_model": JUDGE_MODEL,
        "score_scale": "1-5; higher is better",
        "score_threshold": SCORE_THRESHOLD,
        "answerability_metrics": metrics,
        "mean_scores": {
            name: mean(item["judge"][name] for item in results)
            for name in (
                "correctness",
                "relevance",
                "retrieval_relevance",
                "groundedness",
            )
        },
        "total_examples": len(results),
        "examples": results,
        "failed_answers": [
            item for item in results
            if not item["answer_quality_pass"]
            or not item["answerable_prediction_correct"]
        ],
        "metric_note": (
            "Precision, recall, F1, and the confusion matrix measure retrieval "
            "answerability detection. Mean correctness, relevance, and groundedness "
            "are separate LLM-judge ratings."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_rag_with_judge(), indent=2, ensure_ascii=False))
