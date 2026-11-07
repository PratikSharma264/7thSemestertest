import json

import pandas as pd


def build_prompt(question, rag_answer, reference_df, knowledge_context=None):

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

    knowledge_context = knowledge_context or []
    evidence_records = []
    for index, item in enumerate(knowledge_context, start=1):
        if isinstance(item, dict):
            evidence_records.append({
                "passage": index,
                "source": item.get("source", "N/A"),
                "content": item.get("content", ""),
            })
        else:
            evidence_records.append({
                "passage": index,
                "source": "N/A",
                "content": str(item),
            })
    knowledge_evidence = json.dumps(evidence_records, ensure_ascii=False)

    prompt = f"""
You are an independent evaluator of a Retrieval-Augmented
Generation (RAG) system.

Your task is to determine whether the RAG-generated answer
is sufficient to answer the user's question.

Use both supplied sources for different purposes:
- The labelled reference dataset shows examples of sufficient
  and insufficient answers and calibrates the label distinction.
- The retrieved knowledge-base passages provide the evidence
  for judging the current answer's relevance and coverage.

REFERENCE DATASET:
{dataset_context}

RETRIEVED KNOWLEDGE-BASE PASSAGES:
{knowledge_evidence}

CURRENT QUESTION:
{question}

RAG-GENERATED ANSWER:
{rag_answer}

IMPORTANT INSTRUCTIONS:

1. Judge the RAG answer, not the reference answer.
   The answer may use Markdown, including headings, lists,
   tables, bold text, or citations. Interpret the rendered
   meaning of that content; Markdown syntax is not a defect.
2. Use the reference examples to calibrate completeness and
   relevance. Use the retrieved passages to assess whether the
   current answer addresses the question and is supported by the
   available knowledge-base evidence.
3. A sufficient answer must address the actual question
   and provide enough relevant information.
4. An answer that is merely related to the question is
   not automatically sufficient.
5. An answer that provides only a small portion of the
   requested information should be considered insufficient
   when important information is missing.
6. Do not treat reference answers as evidence for the current
   question; they are examples for label calibration only.
7. Do not invent facts or assume access to documents beyond the
   supplied retrieved passages.
8. Do not use outside knowledge to fill missing information.
9. If the retrieved passages do not contain enough evidence,
   reflect that limitation in the judgement. A truthful or
   appropriately cautious refusal does not, by itself, make the
   answer sufficient.
10. If the user asks for a fact, finding, explanation, estimate,
    comparison, or recommendation and the RAG answer only says
    that the information is unavailable, not reported, or cannot
    be found in the supplied evidence, label it INSUFFICIENT (0). The
    answer has not supplied the requested information, even when
    the retrieved passages genuinely lack that information.
11. An answer that says information is unavailable may be
    SUFFICIENT (1) only when the user explicitly asks whether the
    supplied retrieved passages contain or report that
    information, and the answer directly and accurately answers
    that evidence-availability question using those passages.
12. Judge answer sufficiency separately from whether the RAG
    system behaved responsibly. Do not label a non-answer
    SUFFICIENT merely because it avoided inventing unsupported
    facts. In that case, recognize the grounding as appropriate
    in the reason, but still assign INSUFFICIENT for not answering
    the user's substantive question.
13. Return a binary label:
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
