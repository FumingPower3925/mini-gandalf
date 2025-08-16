#!/usr/bin/env python3

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
)
from typing import Dict, Any, List, Tuple
from collections import Counter
import numpy as np
import os, random, math

def set_seed(seed: int = 42):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def binarize_with_threshold(split, threshold: float = None, is_binary: bool = False):
    def add_labels(ex: Dict[str, Any]) -> Dict[str, Any]:
        sim = ex["similarity"]
        ex["labels"] = int(sim) if is_binary else (1 if sim >= threshold else 0)
        return ex
    return split.map(add_labels)

def counts(preds: np.ndarray, labels: np.ndarray) -> Tuple[int,int,int,int]:
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return tp, fp, tn, fn

def metrics_from_preds(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    tp, fp, tn, fn = counts(preds, labels)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (preds == labels).mean().item()
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    bacc = 0.5 * (tpr + tnr)
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) or 1.0
    mcc = ((tp*tn) - (fp*fn)) / denom
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "bacc": bacc, "mcc": mcc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def tune_threshold(probs_pos: np.ndarray, labels: np.ndarray, metric: str = "mcc") -> Tuple[float, Dict[str, float]]:
    best_t, best_val, best_stats = 0.5, -1e9, {}
    for t in np.linspace(0.05, 0.95, 181):  # step 0.005
        preds = (probs_pos >= t).astype(int)
        stats = metrics_from_preds(preds, labels)
        val = stats.get(metric, 0.0)
        if val > best_val:
            best_val, best_t, best_stats = val, float(t), stats
    return best_t, best_stats

def majority_baseline_metrics(labels: np.ndarray) -> Dict[str, float]:
    maj = 1 if labels.mean() >= 0.5 else 0
    preds = np.full_like(labels, maj)
    return {"majority_label": maj, **metrics_from_preds(preds, labels)}

def train():
    import torch
    set_seed(42)

    ds = load_dataset("Lakera/gandalf_ignore_instructions")
    tr_raw, va_raw, te_raw = ds["train"], ds["validation"], ds["test"]

    sims = tr_raw["similarity"]
    is_binary = set(sims).issubset({0, 1})
    if is_binary:
        threshold = None
        print("Detected binary similarity values in train; using them directly as labels.")
    else:
        env_t = os.getenv("SIMILARITY_LABEL_THRESHOLD")
        if env_t is not None:
            threshold = float(env_t)
            print(f"Using env-provided similarity threshold: {threshold}")
        else:
            s = sorted(sims); mid = len(s)//2
            threshold = s[mid] if len(s)%2==1 else 0.5*(s[mid-1]+s[mid])
            print(f"Using train-fold median similarity threshold: {threshold}")

    tr = binarize_with_threshold(tr_raw, threshold, is_binary)
    va = binarize_with_threshold(va_raw, threshold, is_binary)
    te = binarize_with_threshold(te_raw, threshold, is_binary)
    print("Label balance (train):", Counter(tr["labels"]))
    print("Label balance (val):  ", Counter(va["labels"]))
    print("Label balance (test): ", Counter(te["labels"]))

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tok(batch["text"], padding=False, truncation=True, max_length=256)

    keep = ("text","labels")
    tr_tok = tr.map(tokenize, batched=True, remove_columns=[c for c in tr.column_names if c not in keep])
    va_tok = va.map(tokenize, batched=True, remove_columns=[c for c in va.column_names if c not in keep])
    te_tok = te.map(tokenize, batched=True, remove_columns=[c for c in te.column_names if c not in keep])
    collator = DataCollatorWithPadding(tokenizer=tok)

    cfg = AutoConfig.from_pretrained("distilbert-base-uncased")
    cfg.num_labels = 2
    if hasattr(cfg, "seq_classif_dropout"): cfg.seq_classif_dropout = 0.3
    if hasattr(cfg, "dropout"):             cfg.dropout = 0.2
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", config=cfg)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = softmax(logits)
        preds = (probs[:,1] >= 0.5).astype(int)
        m = metrics_from_preds(preds, labels)
        return {"accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
                "f1": m["f1"], "mcc": m["mcc"], "bacc": m["bacc"]}

    args = TrainingArguments(
        output_dir="./results/input_classifier",
        num_train_epochs=6,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.06,
        weight_decay=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mcc",
        greater_is_better=True,
        logging_dir="./logs/input_classifier",
        logging_steps=10,
        report_to="none",
        seed=42,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tr_tok,
        eval_dataset=va_tok,
        data_collator=collator,
        processing_class=tok,        
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...")
    trainer.train()

    print("\n— Validation @0.5 —")
    va_eval = trainer.evaluate(eval_dataset=va_tok)
    print(va_eval)

    va_pred = trainer.predict(va_tok)
    v_probs = softmax(va_pred.predictions)[:,1]
    v_labels = va_pred.label_ids
    t_mcc, stats_mcc = tune_threshold(v_probs, v_labels, metric="mcc")
    t_f1,  stats_f1  = tune_threshold(v_probs, v_labels, metric="f1")

    print(f"\nVal majority baseline: {majority_baseline_metrics(v_labels)}")
    print(f"Best threshold (MCC): {t_mcc:.3f} -> {stats_mcc}")
    print(f"Best threshold (F1):  {t_f1:.3f}  -> {stats_f1}")

    print("\n— Test @0.5 and @tuned(MCC) —")
    te_pred = trainer.predict(te_tok)
    t_probs = softmax(te_pred.predictions)[:,1]
    t_labels = te_pred.label_ids

    test_maj = majority_baseline_metrics(t_labels)
    test_05  = metrics_from_preds((t_probs >= 0.5).astype(int), t_labels)
    test_tmc = metrics_from_preds((t_probs >= t_mcc).astype(int), t_labels)

    print(f"Test majority baseline: {test_maj}")
    print(f"Test @0.5:            : {test_05}")
    print(f"Test @{t_mcc:.3f} (MCC): {test_tmc}")

    print("\nSaving model...")
    trainer.save_model("./models/input_classifier")
    tok.save_pretrained("./models/input_classifier")
    print("Done.")

if __name__ == "__main__":
    train()
