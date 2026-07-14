### Financial RAG Capstone

#Week 1
-Create the GitHub repo and project structure
-Set up the FastAPI and React frontend
-Set up Docker skeleton with PostgresSQL
-Get DocFinQA downloaded and inspect the data
-Write the loader (gets one example)
-Build the simple retrieval, keyword search
-Add basic evaluation of predicted answer

#Week 2
-Clean up week 1 and make the project runnable
-Build the chunker
-Add embeddings
-Store chunks in ChromaDB
-Build semantic retrieval
-Add the generation step
-Add basic answer evaluation and write Week 2 notes

## Week 2 Chroma indexing

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

Start the API from the `backend` folder:

```bash
uvicorn app.main:app --reload
```

Build a small persisted Chroma index:

```bash
curl -X POST "http://localhost:8000/index-docfinqa?split=train&start_index=0&limit=10&strategy=section&chunk_size=512&overlap=50&reset=true"
```

Query the indexed chunks for a sample DocFinQA question:

```bash
curl "http://localhost:8000/retrieve-chroma-sample?split=train&index=0&top_k=3"
```

Generate an answer with the Week 2-3 RAG pipeline:

```bash
set OPENAI_API_KEY=your_api_key_here
curl -X POST "http://localhost:8000/ask?question=What%20was%20the%20company's%20net%20income%3F&top_k=3"
```

The Chroma store is written to `data/chroma` by default. Set `CHROMA_PERSIST_DIR`
to use a different location.

## Day 7 small retrieval test

Run semantic retrieval on a small DocFinQA sample and save the result file:

```bash
cd backend
python -m app.evaluation.retrieval_eval --index-first --reset --split train --start-index 0 --limit 10 --top-k 3 --output ../data/processed/retrieval_sample_results.json
```

CSV output works too:

```bash
cd backend
python -m app.evaluation.retrieval_eval --split train --start-index 0 --limit 10 --top-k 3 --output ../data/processed/retrieval_sample_results.csv
```

The output includes `question`, `gold_answer`, `retrieved_chunk_ids`, and
`retrieved_text`. It also includes `same_document_hits` and
`top_1_same_document`, which help check whether Chroma is retrieving chunks from
the expected DocFinQA document instead of unrelated indexed examples.
