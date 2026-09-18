from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from src.constants import MODEL_SAMPLE_LIMIT, SCORING_BATCH_SIZE

TARGET_COLUMN = "priority_review"
DISPLAY_TARGET_COLUMN = "priority_label"

# Excluded deliberately: `priority_review` is derived in src/data.py from the
# condition ratings and structural evaluation, so handing those back to the model
# leaks the label. With them included the decision tree scored 0.995 accuracy —
# it was re-deriving its own target, not predicting anything. The term
# `structural_evaluation <= 4` alone recovers 67.5% of all positives.
#
# Kept out of MODEL_FEATURES but still shown in the Bridge Explorer as context.
LEAKING_FEATURES = [
    "structural_evaluation",
    "deck",
    "superstructure",
    "substructure",
]

NUMERIC_FEATURES = [
    "bridge_age",
    "lanes_on_structure",
    "average_daily_traffic",
    "average_daily_truck_traffic",
    "truck_share",
    "skew",
    "number_of_spans_in_main_unit",
    "length_of_maximum_span",
    "structure_length",
    "bridge_roadway_width",
    "deck_width",
    "operating_rating",
    "designated_inspection_frequency",
    "deck_area",
]

CATEGORICAL_FEATURES = [
    "state_name",
    "design_load",
    "type_of_service_on_bridge",
    "type_of_service_under_bridge",
    "kind_of_material_design",
    "type_of_design_construction",
    "scour_critical_bridges",
]

MODEL_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


@dataclass
class ModelBundle:
    dataset: pd.DataFrame
    training_frame: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series
    decision_tree: Pipeline
    random_forest: Pipeline
    metrics: pd.DataFrame
    test_predictions: dict[str, np.ndarray]


def build_model_bundle(frame: pd.DataFrame, random_state: int = 21) -> ModelBundle:
    model_frame = frame.copy()

    if len(model_frame) > MODEL_SAMPLE_LIMIT:
        sampled_parts: list[pd.DataFrame] = []
        per_class_cap = MODEL_SAMPLE_LIMIT // max(model_frame[TARGET_COLUMN].nunique(), 1)
        for _, part in model_frame.groupby(TARGET_COLUMN):
            sampled_parts.append(
                part.sample(
                    n=min(len(part), per_class_cap),
                    random_state=random_state,
                )
            )
        model_frame = pd.concat(sampled_parts, ignore_index=True)

    X = model_frame[MODEL_FEATURES]
    y = model_frame[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    decision_tree = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=5,
                    min_samples_leaf=40,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    random_forest = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=140,
                    max_depth=10,
                    min_samples_leaf=12,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    decision_tree.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)

    test_predictions = {
        "Decision Tree": decision_tree.predict(X_test),
        "Random Forest": random_forest.predict(X_test),
    }

    metrics = pd.DataFrame(
        [
            _metric_row("Decision Tree", y_test, test_predictions["Decision Tree"]),
            _metric_row("Random Forest", y_test, test_predictions["Random Forest"]),
        ]
    )

    return ModelBundle(
        dataset=frame,
        training_frame=model_frame,
        X_test=X_test,
        y_test=y_test,
        decision_tree=decision_tree,
        random_forest=random_forest,
        metrics=metrics,
        test_predictions=test_predictions,
    )


def _metric_row(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | str]:
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def get_pipeline(model_bundle: ModelBundle, model_name: str) -> Pipeline:
    return model_bundle.random_forest if model_name == "Random Forest" else model_bundle.decision_tree


def get_confusion(model_bundle: ModelBundle, model_name: str) -> np.ndarray:
    labels = [0, 1]
    return confusion_matrix(model_bundle.y_test, model_bundle.test_predictions[model_name], labels=labels)


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    feature_names = model.named_steps["prep"].get_feature_names_out()
    raw_importances = model.named_steps["model"].feature_importances_
    totals: dict[str, float] = {}

    for feature_name, importance in zip(feature_names, raw_importances, strict=False):
        clean_name = feature_name.split("__", 1)[-1]
        base_name = clean_name
        for original_feature in [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]:
            if clean_name == original_feature or clean_name.startswith(f"{original_feature}_"):
                base_name = original_feature
                break
        totals[base_name] = totals.get(base_name, 0.0) + float(importance)

    feature_frame = (
        pd.DataFrame({"feature": list(totals.keys()), "importance": list(totals.values())})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return feature_frame


def export_tree_rules(model: Pipeline) -> str:
    feature_names = model.named_steps["prep"].get_feature_names_out()
    tree = model.named_steps["model"]
    cleaned_names = [_friendly_feature_name(name) for name in feature_names]
    return export_text(tree, feature_names=list(cleaned_names), max_depth=4)


def score_state_bridges(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["priority_probability"] = _predict_proba_batched(model, scored)
    scored["predicted_priority"] = np.where(
        scored["priority_probability"] >= 0.5, "Priority Review", "Routine Review"
    )
    return scored.sort_values("priority_probability", ascending=False).reset_index(drop=True)


def _predict_proba_batched(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Score in fixed-size batches so the one-hot matrix never materialises whole.

    The national view is ~469k rows and the encoder produces ~170 columns, so a
    single predict_proba call allocates roughly 640 MB of float64 — on its own
    more than half of Streamlit Community Cloud's 1 GB ceiling. Batching caps
    that at SCORING_BATCH_SIZE rows and leaves the output identical.
    """
    row_count = len(frame)
    if row_count == 0:
        return np.empty(0, dtype=float)

    features = frame[MODEL_FEATURES]
    if row_count <= SCORING_BATCH_SIZE:
        return model.predict_proba(features)[:, 1]

    probabilities = np.empty(row_count, dtype=float)
    for start in range(0, row_count, SCORING_BATCH_SIZE):
        stop = min(start + SCORING_BATCH_SIZE, row_count)
        probabilities[start:stop] = model.predict_proba(features.iloc[start:stop])[:, 1]
    return probabilities


def explain_tree_prediction(model: Pipeline, row: pd.Series) -> list[str]:
    prep = model.named_steps["prep"]
    tree = model.named_steps["model"]
    feature_names = prep.get_feature_names_out()
    transformed = prep.transform(pd.DataFrame([row[MODEL_FEATURES]]))
    node_indicator = tree.decision_path(transformed)
    leaf_id = tree.apply(transformed)[0]
    node_index = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]

    steps: list[str] = []
    for node_id in node_index:
        if node_id == leaf_id:
            value = tree.tree_.value[node_id][0]
            predicted_class = int(np.argmax(value))
            label = "Priority Review" if predicted_class == 1 else "Routine Review"
            steps.append(f"Leaf node reached: the tree predicts `{label}`.")
            continue

        feature_index = tree.tree_.feature[node_id]
        threshold = tree.tree_.threshold[node_id]
        feature_name = _friendly_feature_name(feature_names[feature_index])
        value = transformed[0, feature_index]
        direction = "left" if value <= threshold else "right"
        steps.append(
            f"`{feature_name}` had encoded value `{value:.3f}` compared with threshold `{threshold:.3f}`, "
            f"so the tree moved **{direction}**."
        )
    return steps


def _friendly_feature_name(feature_name: str) -> str:
    return feature_name.replace("num__", "").replace("cat__", "")
