
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# 1. Configuration
SEED = 42
set_seed(SEED)

CSV_PATH = "dataset_A.csv"
MODEL_NAME = (
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)
OUTPUT_DIR = "./biomedbert_sufficiency"
MAX_LENGTH = 512
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 4
WEIGHT_DECAY = 0.01

# 2. Load and validate the dataset
df = pd.read_csv(CSV_PATH)

required = [
    "question", "answer", "sufficiency_label", "source_file"
]
if not set(required).issubset(df.columns):
    raise ValueError(f"Required columns: {required}")

df = df.dropna(subset=required).copy()
df["question"] = df["question"].astype(str).str.strip()
df["answer"] = df["answer"].astype(str).str.strip()
df["source_file"] = df["source_file"].astype(str).str.strip()
df["sufficiency_label"] = df["sufficiency_label"].astype(int)

df = df[
    df["question"].ne("")
    & df["answer"].ne("")
    & df["source_file"].ne("")
].copy()

if not set(df["sufficiency_label"].unique()).issubset({0, 1}):
    raise ValueError("Labels must be binary: 0 or 1.")

# Detect conflicting annotations before removing duplicates.
conflicts = df.groupby(
    ["question", "answer"]
)["sufficiency_label"].nunique()

if (conflicts > 1).any():
    raise ValueError(
        "Conflicting labels for identical question-answer pairs. "
        "Resolve annotations before training."
    )

df = df.drop_duplicates(
    ["question", "answer", "sufficiency_label"]
).reset_index(drop=True)

print("Dataset size:", len(df))
print("Unique questions:", df["question"].nunique())
print("Unique papers:", df["source_file"].nunique())
print(df["sufficiency_label"].value_counts())

# 3. Split by source paper: 70% train, 15% validation, 15% test.
# Source papers must not cross split boundaries.

first_split = GroupShuffleSplit(
    n_splits=1, test_size=0.30, random_state=42
)
train_idx, remaining_idx = next(
    first_split.split(df, groups=df["source_file"])
)

train_df = df.iloc[train_idx].copy()
remaining_df = df.iloc[remaining_idx].copy()

second_split = GroupShuffleSplit(
    n_splits=1, test_size=0.50, random_state=43
)
val_idx, test_idx = next(
    second_split.split(
        remaining_df, groups=remaining_df["source_file"]
    )
)

val_df = remaining_df.iloc[val_idx].copy()
test_df = remaining_df.iloc[test_idx].copy()

assert set(train_df.source_file).isdisjoint(val_df.source_file)
assert set(train_df.source_file).isdisjoint(test_df.source_file)
assert set(val_df.source_file).isdisjoint(test_df.source_file)

# Audit recurring questions across partitions.
question_sets = {
    "train": set(train_df.question),
    "validation": set(val_df.question),
    "test": set(test_df.question),
}
print(
    "Question overlap: train-validation =",
    len(question_sets["train"] & question_sets["validation"])
)
print(
    "Question overlap: train-test =",
    len(question_sets["train"] & question_sets["test"])
)
print(
    "Question overlap: validation-test =",
    len(question_sets["validation"] & question_sets["test"])
)

# 4. Tokenize question + retrieved evidence as a text pair.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class EvidenceDataset(Dataset):
    def __init__(self, frame):
        self.labels = frame["sufficiency_label"].tolist()
        self.encodings = tokenizer(
            frame["question"].tolist(),
            frame["answer"].tolist(),
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: values[idx]
            for key, values in self.encodings.items()
        }
        item["labels"] = self.labels[idx]
        return item

train_ds = EvidenceDataset(train_df)
val_ds = EvidenceDataset(val_df)
test_ds = EvidenceDataset(test_df)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5. Load pretrained encoder + two-class classification head.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "INSUFFICIENT", 1: "SUFFICIENT"},
    label2id={"INSUFFICIENT": 0, "SUFFICIENT": 1},
)

# 6. Evaluate on validation data during fine-tuning.
def metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels, predictions,
            average="binary", pos_label=1,
            zero_division=0,
        )
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_sufficient": precision,
        "recall_sufficient": recall,
        "f1_sufficient": f1,
    }

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_sufficient",
    greater_is_better=True,
    save_total_limit=2,
    logging_steps=20,
    report_to="none",
    seed=SEED,
)

# Print the configured run settings and estimated training steps.
steps_per_epoch = int(np.ceil(len(train_ds) / TRAIN_BATCH_SIZE))
print("\nTraining configuration")
print(f"Model: {MODEL_NAME}")
print(f"Output directory: {OUTPUT_DIR}")
print("Training runs in this script: 1")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE:g}")
print(f"Train batch size/device: {TRAIN_BATCH_SIZE}")
print(f"Eval batch size/device: {EVAL_BATCH_SIZE}")
print(f"Weight decay: {WEIGHT_DECAY}")
print(f"Max sequence length: {MAX_LENGTH}")
print(f"Seed: {SEED}")
print(f"Train/validation/test rows: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
print(f"Estimated optimizer steps/epoch: {steps_per_epoch}")
print(f"Estimated total optimizer steps: {steps_per_epoch * NUM_EPOCHS}")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    processing_class=tokenizer,
    compute_metrics=metrics,
)

# 7. Fine-tune; validation F1 selects the best checkpoint.
trainer.train()

# Save the best checkpoint selected by validation performance.
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 8. Evaluate the selected model ONCE on the held-out test set.
test_output = trainer.predict(test_ds)
test_predictions = np.argmax(
    test_output.predictions, axis=-1
)
test_labels = test_output.label_ids

print("\nFinal held-out test results")
print(
    classification_report(
        test_labels,
        test_predictions,
        labels=[0, 1],
        target_names=["Insufficient", "Sufficient"],
        zero_division=0,
    )
)
print("Confusion matrix:")
print(confusion_matrix(test_labels, test_predictions, labels=[0, 1]))

print("Test accuracy:", accuracy_score(test_labels, test_predictions))

# False-sufficiency rate: insufficient examples predicted sufficient.
tn, fp, fn, tp = confusion_matrix(
    test_labels, test_predictions, labels=[0, 1]
).ravel()

false_sufficiency_rate = fp / (fp + tn) if (fp + tn) else 0.0
print(f"False-sufficiency rate: {false_sufficiency_rate:.4f}")
print("Saved model:", OUTPUT_DIR)
