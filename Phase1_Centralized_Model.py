import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import re
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


ds = pd.read_csv('/Workspace/Users/w10168315@usm.edu/HonorsThesis/final_CLEANED_dataset.csv')
# print(ds)


''' Exploratory Data Analysis '''
# print("Dataset shape:", ds.shape)     # (12985, 5)
# print("\nColumns:", ds.columns.tolist())    # ['text', 'label', 'attack_type', 'is_synthetic', '_text_norm']
# print("\nMissing values:", ds.isnull().sum())    # text: 0, label: 0, attack_type: 0, is_synthetic: 0, _text_norm: 1
# print("\nLabel distribution:", ds['label'].value_counts())    # 0: 8128, 1: 4857 (IMBALANCED DATASET)
# print("\nLabel percentages:", ds['label'].value_counts(normalize=True) * 100)    # 0: 62.59%, 1: 37.40%
# print("\nRemaining exact duplicates:", ds.duplicated(subset=['text']).sum())    # 0



'''Split into Training, Testing, and Validation'''
SEED = 42

# Prefer a group_id assigned when creating encoded/reworded variants. Each group_id must uniquely identify one prompt family.
if "group_id" in ds.columns and ds["group_id"].notna().all():
    groups = ds["group_id"].astype(str)
    print("Using existing group_id.")
else:
    # Fallback only: unique text groups. Does NOT link encoded or reworded versions of the same prompt.
    groups = pd.Series(
        "text_" + ds["text"].astype(str),
        index=ds.index
    )
    print("WARNING: No complete group_id available. Related prompt variants may cross splits.")

ds["split_group"] = groups.values

# Check that groups do not contain conflicting labels
group_conflicts = ds.groupby("split_group")["label"].nunique()

if (group_conflicts > 1).any():
    print("WARNING: Some groups contain both labels. Review these groups before training.")


def choose_split(df, n_splits, target_fraction, seed):
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    candidates = []
    for train_idx, holdout_idx in splitter.split(df, df["label"], groups=df["split_group"]):
        holdout = df.iloc[holdout_idx]
        fraction = len(holdout) / len(df)

        # Prefer target size and similar label distribution
        class_difference = abs(
            holdout["label"].mean() - df["label"].mean()
        )
        score = (abs(fraction - target_fraction) + 0.5 * class_difference)
        candidates.append((score, train_idx, holdout_idx))

    _, train_idx, holdout_idx = min(candidates, key=lambda x: x[0])

    return (df.iloc[train_idx].copy(), df.iloc[holdout_idx].copy())


# 15% test, approximately
train_val, test_df = choose_split(ds, n_splits=7, target_fraction=0.15, seed=SEED)

# Validation should be ~15/85 of remaining data
train_df, val_df = choose_split(
    train_val,
    n_splits=6,
    target_fraction=0.15 / 0.85,
    seed=SEED + 1
)

# Reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Verify that groups do not overlap
train_groups = set(train_df["split_group"])
val_groups = set(val_df["split_group"])
test_groups = set(test_df["split_group"])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

assert len(train_df) + len(val_df) + len(test_df) == len(ds)

# Check class coverage
for name, frame in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    assert frame["label"].nunique() == 2, (
        f"{name} does not contain both labels."
    )

print("\nDataset splits:")
for name, frame in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    print(
        f"{name}: {len(frame)} rows "
        f"({len(frame)/len(ds):.1%}), "
        f"attack rate: {frame['label'].mean():.1%}"
    )

# Save each split
train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("\nAll three splits saved successfully.")


### Dataset splits: ###
# Train: 9275 rows (71.4%), attack rate: 37.4%
# Validation: 1855 rows (14.3%), attack rate: 37.9%
# Test: 1855 rows (14.3%), attack rate: 36.8%





''' Shared evaluation functions '''

def evaluate_predictions(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "FPR": fp / (fp + tn) if (fp + tn) else 0.0
    }


def select_threshold(y_true, probabilities):
    # Optimize F1 on validation data only
    thresholds = np.arange(0.10, 0.91, 0.01)

    scores = [
        f1_score(
            y_true,
            (probabilities >= t).astype(int),
            zero_division=0
        )
        for t in thresholds
    ]
    return float(thresholds[np.argmax(scores)])

validation_results = {}
test_probabilities = {}
selected_thresholds = {}





'''Model 1 — Rule-based prompt-injection detection'''

attack_patterns = [
    r"\bignore (all |any |your )?(previous|prior) instructions\b",
    r"\bdisregard (all |your )?(previous|prior) instructions\b",
    r"\breveal (your |the )?(system|developer) prompt\b",
    r"\bshow (me )?(your |the )?hidden instructions\b",
    r"\bbypass (the |your )?(safety|security) (rules|filters|restrictions)\b",
    r"\byou are now (dan|in developer mode)\b",
    r"\bforget (all |your )?(previous|prior) instructions\b",
    r"\bdo not follow (your |the )?(original|system|developer) instructions\b"
]

compiled_patterns = [
    re.compile(p, re.IGNORECASE)
    for p in attack_patterns
]

def rule_based_predict(text):
    return int(any(
        pattern.search(str(text))
        for pattern in compiled_patterns
    ))

rule_val_pred = val_df["text"].apply(rule_based_predict).to_numpy()
validation_results["Rule-based"] = evaluate_predictions(val_df["label"], rule_val_pred)
print("Rule-based validation results:", validation_results["Rule-based"])

''' OUTPUT of Rule-based prompt-injection detection:
{'Accuracy': 0.6366576819407008, 'Precision': 1.0, 'Recall': 0.041251778093883355, 'F1': 0.07923497267759563, 'FPR': np.float64(0.0)} '''






''' MODEL 2: TF-IDF + Logistic Regression'''

tfidf_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=SEED
    ))
])

# Train using training data only
tfidf_model.fit(
    train_df["text"],
    train_df["label"]
)

# Validation probabilities
tfidf_val_probs = tfidf_model.predict_proba(
    val_df["text"]
)[:, 1]

# Select threshold using validation data
tfidf_threshold = select_threshold(
    val_df["label"], tfidf_val_probs
)

selected_thresholds["TF-IDF + LR"] = tfidf_threshold

tfidf_val_pred = (
    tfidf_val_probs >= tfidf_threshold
).astype(int)

validation_results["TF-IDF + LR"] = evaluate_predictions(
    val_df["label"], tfidf_val_pred
)

print("Selected threshold:", tfidf_threshold)
print("TF-IDF + Logistic Regression: ", validation_results["TF-IDF + LR"])


'''
### OUTPUT ###
Selected threshold: 0.3699999999999999
TF-IDF + Logistic Regression:  {'Accuracy': 0.8964959568733154, 'Precision': 0.8016528925619835, 'Recall': 0.9658605974395448, 'F1': 0.8761290322580645, 'FPR': np.float64(0.14583333333333334)}
'''




''' Model 3 — Sentence embeddings + Logistic Regression
Converts prompts into numerical representations of their meaning & trains a classifier on those representations '''

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Generate embeddings
X_train_emb = embedding_model.encode(
    train_df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

X_val_emb = embedding_model.encode(
    val_df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

# Train classifier
embedding_classifier = LogisticRegression(
    max_iter=2000,
    random_state=SEED
)

embedding_classifier.fit(
    X_train_emb,
    train_df["label"]
)

# Validation probabilities
emb_val_probs = embedding_classifier.predict_proba(
    X_val_emb
)[:, 1]

emb_threshold = select_threshold(
    val_df["label"], emb_val_probs
)

selected_thresholds["Embeddings + LR"] = emb_threshold

emb_val_pred = (
    emb_val_probs >= emb_threshold
).astype(int)

validation_results["Embeddings + LR"] = evaluate_predictions(
    val_df["label"], emb_val_pred
)

print("Selected threshold:", emb_threshold)
print("Sentence embeddings + Logistic Regression: ", validation_results["Embeddings + LR"])

'''
# OUTPUT
Selected threshold: 0.43999999999999984
Sentence embeddings + Logistic Regression:  {'Accuracy': 0.8776280323450135, 'Precision': 0.8148148148148148, 'Recall': 0.8762446657183499, 'F1': 0.8444139821795751, 'FPR': np.float64(0.12152777777777778)}
'''