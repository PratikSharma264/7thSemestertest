import os
import json

from dotenv import load_dotenv

from sufficiency_judge.dataset_loader import load_dataset, prepare_reference_data

from sufficiency_judge.prompt_builder import build_prompt


load_dotenv()

client = None


def judge_sufficiency(question, rag_answer, knowledge_context=None, dataset=None):
    """Judge an answer using reference examples and retrieved KB passages."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    if not isinstance(rag_answer, str) or not rag_answer.strip():
        raise ValueError("rag_answer must be a non-empty string.")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to the environment or .env file.")

    model = os.getenv("GEMINI_MODEL")
    if not model:
        raise RuntimeError("GEMINI_MODEL is not set in the .env file.")

    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise RuntimeError(
            "Gemini SDK is not installed in the Python environment running this app. "
            "Install it with `python -m pip install google-genai` using the same "
            "Python interpreter that runs query.py."
        ) from error

    if dataset is None:
        dataset = load_dataset()

    global client
    if client is None:
        client = genai.Client(api_key=api_key)

    reference_df = prepare_reference_data(
        dataset,
        question
    )

    prompt = build_prompt(
        question,
        rag_answer,
        reference_df,
        knowledge_context=knowledge_context,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0
        )
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")
    result = json.loads(response.text)

    if not isinstance(result, dict) or not {"sufficiency_label", "judgement", "reason"}.issubset(result):
        raise ValueError("Gemini response is missing required judgement fields.")

    label = int(result["sufficiency_label"])

    if label not in (0, 1):
        raise ValueError("Invalid sufficiency label.")

    expected_judgement = (
        "SUFFICIENT" if label == 1 else "INSUFFICIENT"
    )

    if result["judgement"] != expected_judgement:
        raise ValueError("Inconsistent judgement and label.")

    return {
        "gemini_label": label,
        "gemini_judgement": expected_judgement,
        "gemini_reason": result["reason"]
    }


if __name__ == "__main__":

    question = "What are the risk factors for type 2 diabetes in adolescents?"

    rag_answer = (
        "Obesity is one of the major risk factors "
        "for type 2 diabetes in adolescents."
    )

    result = judge_sufficiency(
        question,
        rag_answer,
    )

    print(json.dumps(result, indent=4))
