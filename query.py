
"""
Query the Diabetes Knowledge Base using MedCPT and Groq.

After generating the RAG answer, use the fine-tuned
BioMedBERT classifier to evaluate answer sufficiency.
"""

import chromadb
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import requests
from dotenv import load_dotenv
import os

# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"

# Path to your fine-tuned BioMedBERT model
CLASSIFIER_MODEL_PATH = "./biomedbert_sufficiency"

MAX_LENGTH = 512

EVALUATOR_CLASSIFIER = "1"
EVALUATOR_LLM_JUDGE = "2"

classifier_tokenizer = None
classifier_model = None

# ==========================================
# DEVICE CONFIGURATION
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# ==========================================
# STEP 1: LOAD MEDCPT QUERY ENCODER
# ==========================================

print("\nLoading MedCPT Query Encoder...")

query_tokenizer = AutoTokenizer.from_pretrained(
    QUERY_MODEL_NAME
)

query_model = AutoModel.from_pretrained(
    QUERY_MODEL_NAME
).to(device)

query_model.eval()

print("MedCPT Query Encoder loaded successfully!")

# ==========================================
# STEP 2: LOAD BIOMEDBERT CLASSIFIER
# ==========================================

def load_classifier():
    """Load the local classifier on demand."""
    global classifier_tokenizer, classifier_model
    if classifier_model is None:
        print("\nLoading BioMedBERT Sufficiency Classifier...")
        classifier_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_PATH)
        classifier_model = AutoModelForSequenceClassification.from_pretrained(
            CLASSIFIER_MODEL_PATH
        ).to(device)
        classifier_model.eval()
        print("BioMedBERT Classifier loaded successfully!")

# ==========================================
# STEP 3: CONNECT TO CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path="./vector_db"
)

collection = client.get_collection(
    name="research_abstracts"
)

print(
    f"\nConnected to collection: 'research_abstracts' "
    f"({collection.count()} documents)"
)

# ==========================================
# STEP 4: MEDCPT QUERY ENCODING
# ==========================================

def encode_query(question):

    encoded = query_tokenizer(
        [question],
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=512,
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        output = query_model(**encoded)

    query_embedding = (
        output.last_hidden_state[:, 0, :]
        .squeeze(0)
        .cpu()
        .numpy()
        .tolist()
    )

    return query_embedding


# ==========================================
# STEP 5: SEARCH CHROMADB
# ==========================================

def search(question, top_k=3):

    query_embedding = encode_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return results


# ==========================================
# STEP 6: DISPLAY RETRIEVED DOCUMENTS
# ==========================================

def display_results(question, results):

    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc_id, doc, meta, dist) in enumerate(
        zip(ids, docs, metas, distances),
        start=1
    ):

        similarity = 1 - dist

        print(f"\n--- Result {rank} ---")
        print(f"Source: {meta.get('source', 'N/A')}")
        print(f"Similarity: {similarity:.4f}")
        print("Content:")
        print(doc[:300])

        if len(doc) > 300:
            print(
                f"[...{len(doc) - 300} more characters]"
            )

    print("\n" + "=" * 60)


def display_source_files(results):

    return [
        meta.get("source", "N/A")
        for meta in results["metadatas"][0]
    ]


# ==========================================
# STEP 7: GENERATE RAG RESPONSE USING GROQ
# ==========================================

def generate_rag_response(question, results, api_key):

    if not api_key:
        return (
            "Groq API key not set. "
            "Please configure GROQ_API_KEY in your .env file."
        )

    print("\nGenerating response with Groq...")

    docs = results["documents"][0]

    context = "\n\n".join(
        [
            f"Document {i+1}:\n{doc}"
            for i, doc in enumerate(docs)
        ]
    )

    system_prompt = (
        "You are a helpful medical assistant specializing "
        "in diabetes for children and adolescents. "
        "Use ONLY the provided context to answer the user's question. "
        "Do not introduce unsupported facts. "
        "If the answer cannot be found in the context, say "
        "'I cannot find the answer in the provided documents.' "
        "Clearly distinguish reported findings from interpretations."
    )

    user_prompt = (
        f"Context information:\n"
        f"---------------------\n"
        f"{context}\n"
        f"---------------------\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.2
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]

    except requests.exceptions.HTTPError as e:

        return (
            f"Error communicating with Groq API: {e}\n"
            f"Details: {e.response.text}"
        )

    except Exception as e:

        return f"Error communicating with Groq API: {e}"


# ==========================================
# STEP 8: CLASSIFY RAG ANSWER SUFFICIENCY
# ==========================================

def classify_sufficiency(question, rag_answer):

    """
    Classify whether the generated RAG answer sufficiently
    answers the user's question.

    Input:
        question: Original user question.
        rag_answer: Complete generated RAG response.

    Output:
        Predicted label and class probabilities.
    """

    if not rag_answer or rag_answer.startswith("Error communicating"):
        return None

    load_classifier()

    # Encode question-answer pair.
    # Markdown tables, lists and formatting are accepted as text.
    inputs = classifier_tokenizer(
        question,
        rag_answer,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = classifier_model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    prediction = torch.argmax(probabilities).item()

    labels = {
        0: "INSUFFICIENT",
        1: "SUFFICIENT"
    }

    return {
        "label": labels[prediction],
        "prediction": prediction,
        "insufficient_probability": probabilities[0].item(),
        "sufficient_probability": probabilities[1].item(),
        "confidence": probabilities[prediction].item()
    }


def judge_sufficiency(question, rag_answer, results):
    """Judge answer sufficiency using labels plus retrieved KB passages."""
    from sufficiency_judge.llm_judge import judge_sufficiency as run_judge

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    knowledge_context = [
        {
            "source": metadata.get("source", "N/A"),
            "content": document,
        }
        for document, metadata in zip(documents, metadatas)
    ]

    return run_judge(
        question,
        rag_answer,
        knowledge_context=knowledge_context,
    )


def choose_evaluator():
    """Ask which sufficiency evaluator to use for this session."""
    while True:
        print("\nChoose a sufficiency evaluator:")
        print("1. Classification Model (BioMedBERT)")
        print("2. LLM Sufficiency Judge (Gemini)")
        choice = input("Enter 1 or 2: ").strip()
        if choice in (EVALUATOR_CLASSIFIER, EVALUATOR_LLM_JUDGE):
            return choice
        print("Please enter 1 or 2.")


# ==========================================
# STEP 9: DISPLAY SUFFICIENCY RESULTS
# ==========================================

def display_sufficiency(result):

    print("\n" + "=" * 60)
    print("EVIDENCE SUFFICIENCY CLASSIFICATION")
    print("=" * 60)

    if result is None:

        print("Classification skipped.")
        return

    print(f"Predicted Label: {result['label']}")

    print(
        f"Classification Confidence: "
        f"{result['confidence']:.2%}"
    )

    print(
        f"Insufficient Probability: "
        f"{result['insufficient_probability']:.2%}"
    )

    print(
        f"Sufficient Probability: "
        f"{result['sufficient_probability']:.2%}"
    )

    print("=" * 60)


def display_judge_result(result):
    print("\n" + "=" * 60)
    print("LLM SUFFICIENCY JUDGEMENT")
    print("=" * 60)
    if result is None:
        print("Judgement skipped.")
    else:
        print(f"Predicted Label: {result['gemini_judgement']}")
        print(f"Sufficiency Label: {result['gemini_label']}")
        print(f"Reason: {result['gemini_reason']}")
    print("=" * 60)


# ==========================================
# STEP 10: INTERACTIVE QUERY LOOP
# ==========================================

if __name__ == "__main__":

    print("\n" + "-" * 60)
    print("DIABETES KNOWLEDGE BASE — RAG + SUFFICIENCY")
    print("Type 'quit' or 'exit' to stop.")
    print("-" * 60)

    evaluator = choose_evaluator()
    print("Using: " + ("BioMedBERT Classification Model" if evaluator == EVALUATOR_CLASSIFIER else "Gemini LLM Sufficiency Judge"))

    while True:

        print()

        question = input("Your question: ").strip()

        if not question:
            print("Please enter a question.")
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        # 1. Retrieve relevant documents
        results = search(question, top_k=3)

        # 2. Display retrieved documents
        display_results(question, results)

        # 3. Generate RAG answer
        print("\n" + "=" * 60)
        print("LLM ANSWER (RAG)")
        print("-" * 60)

        answer = generate_rag_response(
            question,
            results,
            GROQ_API_KEY
        )

        print(answer)

        # 4. Evaluate the RAG answer using the selected evaluator.
        if evaluator == EVALUATOR_CLASSIFIER:
            display_sufficiency(classify_sufficiency(question, answer))
        else:
            try:
                display_judge_result(judge_sufficiency(question, answer, results))
            except Exception as error:
                print(f"LLM sufficiency judgement failed: {error}")

        # 6. Display retrieved source files
        print("\nSOURCE FILES:")
        print(display_source_files(results))

        print("\n")
