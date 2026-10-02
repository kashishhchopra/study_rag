"""
Runs a fixed set of test questions through the RAG pipeline and prints the
results, so you can eyeball answer quality and source grounding.

Run with:
    python test_questions.py

Requires the index to already be built (`python ingest.py`).
"""

from query import generate_answer

TEST_QUESTIONS = [
    "What is the difference between bias and variance in a model?",
    "Why is ReLU commonly preferred over sigmoid in hidden layers?",
    "What is the key difference between BERT and GPT pretraining objectives?",
    "What problem does Retrieval-Augmented Generation solve?",
    "What is the difference between semantic segmentation and instance segmentation?",
    "What is L1 regularization also known as, and what effect does it have on coefficients?",
    "What is the capital of France?",  # not in notes -> should trigger the "I don't know" refusal
]


def run():
    print(f"Running {len(TEST_QUESTIONS)} test questions...\n")
    for i, q in enumerate(TEST_QUESTIONS, 1):
        result = generate_answer(q)
        print(f"{i}. Q: {result['question']}")
        print(f"   A: {result['answer']}")
        sources = ", ".join(f"{s['source']} ({s['score']})" for s in result["sources"])
        print(f"   Sources: {sources or 'none'}")
        print("-" * 80)


if __name__ == "__main__":
    run()
