from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle
import os
from training_data import TRAINING_DATA

CATEGORIES = [
    "factual_knowledge",
    "mathematical_reasoning",
    "sentiment_classification",
    "text_summarization",
    "named_entity_recognition",
    "code_debugging",
    "logical_reasoning",
    "code_generation",
]


def main():
    X = [item["prompt"] for item in TRAINING_DATA]
    y = [item["label"] for item in TRAINING_DATA]

    print(f"\nDataset summary:")
    for label in CATEGORIES:
        print(f"  {label:28s}: {y.count(label)} examples")
    print(f"  {'TOTAL':28s}: {len(y)} examples\n")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            strip_accents="unicode",
            lowercase=True,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ))
    ])

    print("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"Cross-validation accuracy: {cv_scores.mean():.1%} +/- {cv_scores.std():.1%}")
    print(f"Per-fold scores: {[f'{s:.1%}' for s in cv_scores]}\n")

    pipeline.fit(X, y)

    # Edge cases spanning categories that could plausibly be confused with
    # each other (e.g. factual_knowledge vs mathematical_reasoning, or
    # code_debugging vs code_generation).
    edge_cases = [
        ("What is 25% of 180?", "mathematical_reasoning"),
        ("What is gravity?", "factual_knowledge"),
        ("Fix the bug in this sorting function", "code_debugging"),
        ("Write a function to sort a list", "code_generation"),
        ("Extract all person names from this paragraph", "named_entity_recognition"),
        ("Is this review positive or negative?", "sentiment_classification"),
        ("Summarize this in two sentences", "text_summarization"),
        ("If all cats are mammals and all mammals are animals, are all cats animals?", "logical_reasoning"),
        ("Explain how photosynthesis works", "factual_knowledge"),
        ("A train travels 150 miles in 3 hours, what is its speed?", "mathematical_reasoning"),
    ]

    print("Edge case checks:")
    passed = 0
    for prompt, expected in edge_cases:
        predicted = pipeline.predict([prompt])[0]
        status = "PASS" if predicted == expected else "FAIL"
        if predicted == expected:
            passed += 1
        print(f"  [{status}] expected={expected:28s} got={predicted:28s} '{prompt[:50]}'")

    print(f"\n{passed}/{len(edge_cases)} edge cases passed.\n")

    out_path = os.path.join(os.path.dirname(__file__), "router_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(pipeline, f)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Model saved: {out_path} ({size_kb:.0f} KB)\n")

    return cv_scores.mean(), passed, len(edge_cases)


if __name__ == "__main__":
    main()
