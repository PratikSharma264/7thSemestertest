# Diabetes RAG

A retrieval-augmented question-answering project for a diabetes research
knowledge base. It uses MedCPT embeddings with ChromaDB, Groq to generate
answers from retrieved passages, and either a local BioMedBERT classifier or
the Gemini-based judge to assess answer sufficiency.

## Requirements

- Python 3.10 or later
- A Groq API key for answer generation
- The project knowledge-base `.txt` files
- For classifier evaluation, the fine-tuned `model.safetensors` file
- For Gemini evaluation, a Gemini API key and model name

The model weights, training checkpoints, knowledge-base documents, and
generated ChromaDB database are kept out of Git. You need to supply the
knowledge base and classifier weights separately.

## Setup (Windows PowerShell)

Run these commands from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, allow scripts for the
current terminal session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Put the source `.txt` documents in `knowledge_base\`. The ingestion script
reads top-level `.txt` files from that directory.

Create a `.env` file in the repository root:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Before generating an answer, Groq checks the retrieved passages for relevance.
Only passages it accepts are used for the answer, sent to the Gemini judge, or
shown in the source list. If none are relevant or the check fails, the app
abstains and shows no source files.

For the Gemini sufficiency judge, also add:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_gemini_model_name
```

Do not commit `.env` or put API keys in source files.

## Prepare the knowledge base

Run ingestion from the repository root:

```powershell
python ingest.py
```

This downloads/loads the MedCPT Article Encoder and creates the persistent
ChromaDB collection `research_abstracts` under `vector_db\`. **Ingestion
deletes and recreates that collection before loading the documents**, so
re-running it replaces the existing collection.

## Run the interactive RAG application

For classifier-based sufficiency evaluation (choice **1** or **3**), place
the fine-tuned `model.safetensors` in `biomedbert_sufficiency\`, alongside the tracked
`config.json` and tokenizer files. Obtain this weights file from the project’s
training output; it is not included in the repository.

Then run:

```powershell
python query.py
```

Choose **1** for the local BioMedBERT classifier, **2** for the Gemini
sufficiency judge, or **3** to run both evaluators on the same RAG answer and
retrieved passages. The app uses Groq for RAG answer generation, loads MedCPT
from Hugging Face on first use, and requires the `research_abstracts` collection
created during ingestion.




Other useful commands, run from the repository root:

```powershell
python test_medcpt.py
python verify_embedding_storage.py
python validate_chunks.py
python inspect_db.py
```

`validate_chunks.py` writes a chunk report and chart in the repository root.
