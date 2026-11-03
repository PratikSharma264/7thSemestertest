import json

import pandas as pd


def build_prompt(question, rag_answer, reference_df):

    # Keep requests compact while giving the judge examples of both classes.
    # The full reference CSV can exceed practical request limits.
    examples_per_label = 8
    sampled_groups = [
        group.sample(
            n=min(examples_per_label, len(group)),
            random_state=0,
        )
        for _, group in reference_df.groupby("sufficiency_label")
    ]
    sampled = (
        pd.concat(sampled_groups)
        if sampled_groups
        else reference_df.head(0)
    )

    records = sampled[
        [
            "question",
            "answer",
            "sufficiency_label",
            "source_file"
        ]
    ].to_dict(orient="records")

    dataset_context = json.dumps(
        records,
        ensure_ascii=False
    )

    prompt = f"""
You are an independent evaluator of a Retrieval-Augmented
Generation (RAG) system.

Your task is to determine whether the RAG-generated answer
is sufficient to answer the user's question.

You must use the supplied labelled reference dataset
to understand the distinction between sufficient and
insufficient answers.

REFERENCE DATASET:
{dataset_context}

CURRENT QUESTION:
{question}

RAG-GENERATED ANSWER:
{rag_answer}

IMPORTANT INSTRUCTIONS:

1. Judge the RAG answer, not the reference answer.
   The answer may use Markdown, including headings, lists,
   tables, bold text, or citations. Interpret the rendered
   meaning of that content; Markdown syntax is not a defect.
2. Use the reference dataset to understand the expected
   level of completeness and relevance.
3. A sufficient answer must address the actual question
   and provide enough relevant information.
4. An answer that is merely related to the question is
   not automatically sufficient.
5. An answer that provides only a small portion of the
   requested information should be considered insufficient
   when important information is missing.
6. Do not assume that information is correct merely
   because it appears in the reference dataset.
7. Do not invent facts or assume access to retrieved
   documents.
8. Do not use outside knowledge to fill missing information.
9. If the answer says that the information is unavailable
   or cannot be found, judge whether that response actually
   answers the question.
10. Return a binary label:
    1 = Sufficient
    0 = Insufficient

Do not simply copy a label from a reference record.
Make an independent judgement of the current RAG answer.

Return JSON only, using this structure:

{{
    "sufficiency_label": 0,
    "judgement": "INSUFFICIENT",
    "reason": "Brief explanation of the judgement."
}}
"""
    return prompt
