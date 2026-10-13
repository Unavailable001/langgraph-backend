
import re
from typing import Any

from dotenv import load_dotenv
from langsmith import Client

from appRag.main import run_rag

load_dotenv()

client = Client()
DATASET_NAME = "Java RAG Evaluation"


# --------------------------------------------------
# 1. DATASET
# Add the ACTUAL relevant chunk IDs for each question.
# Empty lists allow execution but cannot provide
# meaningful retrieval precision, recall, or F1.
# --------------------------------------------------

examples = [
    {
        "inputs": {
            "question": "What is Java?"
        },
        "outputs": {
            "answer": (
                "Java is a high-level, object-oriented "
                "programming language."
            ),
            "relevant_chunk_ids": [],  # Fill in actual IDs
        },
    },
    {
        "inputs": {
            "question": "What are the main features of Java?"
        },
        "outputs": {
            "answer": (
                "Java provides object-oriented programming, "
                "platform independence, security, robustness, "
                "and automatic memory management."
            ),
            "relevant_chunk_ids": [],  # Fill in actual IDs
        },
    },
]


# --------------------------------------------------
# 2. CREATE OR SYNCHRONIZE DATASET
# Existing examples are updated instead of ignored.
# --------------------------------------------------

def ensure_dataset(client: Client) -> None:
    dataset = next(
        (
            item
            for item in client.list_datasets()
            if item.name == DATASET_NAME
        ),
        None,
    )

    if dataset is None:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Evaluation dataset for Java RAG",
        )

    existing_examples = list(
        client.list_examples(dataset_id=dataset.id)
    )

    examples_by_question = {
        item.inputs.get("question"): item
        for item in existing_examples
    }

    for item in examples:
        question = item["inputs"]["question"]
        existing = examples_by_question.get(question)

        if existing is None:
            client.create_example(
                inputs=item["inputs"],
                outputs=item["outputs"],
                dataset_id=dataset.id,
            )
        else:
            client.update_example(
                example_id=existing.id,
                inputs=item["inputs"],
                outputs=item["outputs"],
            )

    print(f"Dataset ready: {DATASET_NAME}")


# --------------------------------------------------
# 3. NORMALIZE TEXT
# --------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------
# 4. RUN YOUR RAG PIPELINE
# Expected dictionary:
# {
#     "answer": "...",
#     "retrieved_chunk_ids": ["id1", "id2"]
# }
# --------------------------------------------------

def predict(inputs: dict) -> dict:
    question = inputs["question"]
    result = run_rag(question)

    if isinstance(result, str):
        answer = result
        retrieved_chunk_ids = []

    elif isinstance(result, dict):
        answer = result["answer"]
        retrieved_chunk_ids = result.get(
            "retrieved_chunk_ids", []
        )

    else:
        raise TypeError(
            "run_rag() must return a string or a dictionary "
            "containing an 'answer' key."
        )

    return {
        "answer": answer,
        "retrieved_chunk_ids": retrieved_chunk_ids,
    }


# --------------------------------------------------
# 5. ANSWER ACCURACY
# Exact match after normalization.
# --------------------------------------------------

def answer_accuracy(
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    predicted = normalize_text(outputs["answer"])
    expected = normalize_text(reference_outputs["answer"])

    return {
        "key": "answer_accuracy",
        "score": float(predicted == expected),
    }


# --------------------------------------------------
# 6. RETRIEVAL PRECISION
# TP / (TP + FP)
# --------------------------------------------------

def retrieval_precision(
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    predicted = set(outputs.get("retrieved_chunk_ids", []))
    relevant = set(reference_outputs.get("relevant_chunk_ids", []))

    tp = len(predicted & relevant)
    fp = len(predicted - relevant)

    score = tp / (tp + fp) if tp + fp else 0.0

    return {
        "key": "retrieval_precision",
        "score": score,
    }


# --------------------------------------------------
# 7. RETRIEVAL RECALL
# TP / (TP + FN)
# --------------------------------------------------

def retrieval_recall(
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    predicted = set(outputs.get("retrieved_chunk_ids", []))
    relevant = set(reference_outputs.get("relevant_chunk_ids", []))

    tp = len(predicted & relevant)
    fn = len(relevant - predicted)

    score = tp / (tp + fn) if tp + fn else 0.0

    return {
        "key": "retrieval_recall",
        "score": score,
    }


# --------------------------------------------------
# 8. RETRIEVAL F1 SCORE
# Harmonic mean of precision and recall.
# --------------------------------------------------

def retrieval_f1(
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    predicted = set(outputs.get("retrieved_chunk_ids", []))
    relevant = set(reference_outputs.get("relevant_chunk_ids", []))

    tp = len(predicted & relevant)
    fp = len(predicted - relevant)
    fn = len(relevant - predicted)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "key": "retrieval_f1",
        "score": f1,
    }


# --------------------------------------------------
# 9. RUN LANGSMITH EVALUATION
# --------------------------------------------------

def evaluate_rag() -> dict[str, str]:
    """Run the LangSmith evaluation and return an API-serializable summary."""
    ensure_dataset(client)

    results = client.evaluate(
        predict,
        data=DATASET_NAME,
        evaluators=[
            answer_accuracy,
            retrieval_precision,
            retrieval_recall,
            retrieval_f1,
        ],
        experiment_prefix="java-rag-evaluation",
        description=(
            "Evaluate answer accuracy and retrieval metrics "
            "for the Java RAG application."
        ),
    )

    return {
        "experiment_name": results.experiment_name,
        "results": str(results),
    }


def main() -> None:
    summary = evaluate_rag()
    print("Experiment:", summary["experiment_name"])
    print("Results:", summary["results"])


if __name__ == "__main__":
    main()