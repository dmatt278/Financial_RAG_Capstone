from app.data_loader import load_docfinqa_example
from fastapi import FastAPI
from app.retriever import retrieve_top_k_chunks
from app.evaluation import evaluate_answer

app = FastAPI(title='Financial RAG Capstone API')

#checks that the API is running
@app.get("/")
def root():
    return {
        "message": "Financial RAG backend is running"
    }

#health check
@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }

#placeholder question-answer endpoint
@app.post("/ask")
def ask_questions(question: str):
    return {
        "question": question,
        "answer": "This is a placeholder answer.",
        "sources": []
    }

@app.get("/sample-docfinqa")
def sample_docfinqa():
    example = load_docfinqa_example(split="train", index=0)

    return {
        "question_id": example["question_id"],
        "question": example["question"],
        "gold_answer": example["gold_answer"],
        "program": example["program"],
        "document_length": len(example["document_text"]),
        "document_preview": example["document_text"][:500],
    }

@app.get("/retrieve-sample")
def retrieve_sample():
    example = load_docfinqa_example(split="train", index=0)

    retrieved_chunks = retrieve_top_k_chunks(
        question=example["question"],
        document_text=example["document_text"],
        top_k=3
    )

    return {
        "question_id": example["question_id"],
        "question": example["question"],
        "gold_answer": example["gold_answer"],
        "retrieved_chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"],
                "preview": chunk["text"][:500]
            }
            for chunk in retrieved_chunks
        ]
    }

@app.get("/evaluate-sample")
def evaluate_sample():
    example = load_docfinqa_example(split="train", index=0)

    # Temporary fake prediction for Week 1
    # Later this will come from the LLM
    predicted_answer = example["gold_answer"]

    evaluation = evaluate_answer(
        predicted_answer=predicted_answer,
        gold_answer=example["gold_answer"]
    )

    return {
        "question_id": example["question_id"],
        "question": example["question"],
        "evaluation": evaluation,
    }