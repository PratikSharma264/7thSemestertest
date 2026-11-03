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

For classifier-based sufficiency evaluation, place the fine-tuned
`model.safetensors` in `biomedbert_sufficiency\`, alongside the tracked
`config.json` and tokenizer files. Obtain this weights file from the project’s
training output; it is not included in the repository.

Then run:

```powershell
python query.py
```

Choose **1** for the local BioMedBERT classifier or **2** for the Gemini
sufficiency judge. Both options use Groq for RAG answer generation. The app
loads MedCPT from Hugging Face on first use and requires the `research_abstracts`
collection created during ingestion.

## Generate the testcase answers

`testing\model_testcases.csv` contains the questions to process. With the
knowledge base ingested and `GROQ_API_KEY` set, run:

```powershell
python testing\run_model_testcases.py
```

The script writes each generated answer and its sufficiency label (`1` =
sufficient, `0` = insufficient) to the CSV as it goes. It can be rerun to
resume after an interruption. This uses the local BioMedBERT classifier, so
the fine-tuned `model.safetensors` must be available as described above.


Other useful commands, run from the repository root:

```powershell
python test_medcpt.py
python verify_embedding_storage.py
python validate_chunks.py
python inspect_db.py
```

`validate_chunks.py` writes a chunk report and chart in the repository root.
