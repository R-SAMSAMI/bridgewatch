# BridgeWatch

Interactive infrastructure analytics dashboard for triaging bridge inspection priority across the U.S. bridge inventory.

**[Open the live app](https://rei-bridgewatch.streamlit.app/)** - all 469,434 bridges, no setup.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rei-bridgewatch.streamlit.app/)

## Demo

![BridgeWatch demo](assets/media/demo-preview.gif)

Full recording (64s, 1.4 MB): [View the demo video](assets/media/demo-recording.mp4)

## Overview

BridgeWatch is a Streamlit dashboard built around the Federal Highway Administration's National Bridge Inventory. It helps users explore bridge condition patterns, compare tree-based models, and inspect bridge-level explanations for why an asset is flagged for closer review.

The app helps users:

- filter the national bridge inventory by state
- compare an interpretable decision tree with a random forest benchmark
- review bridges with the highest predicted inspection priority
- inspect feature importance and tree rules
- walk through bridge-level predictions with an explainable path view

## Product Highlights

- National bridge data workflow with state-by-state exploration
- Decision Tree and Random Forest comparison in one focused interface
- Bridge-level explanation panel for interpretable triage decisions
- Clean dashboard layout with media assets ready for GitHub presentation

## Model Approach

BridgeWatch frames the problem as **Priority Review** versus **Routine Review** using inspection-relevant bridge attributes from the National Bridge Inventory.

The modeling workflow includes:

- a `DecisionTreeClassifier` for interpretable split logic
- a `RandomForestClassifier` for a stronger ensemble benchmark
- numeric and categorical preprocessing through a scikit-learn pipeline
- feature importance and confusion matrix views for model inspection

### Leakage audit

The `priority_review` label is constructed from the condition ratings, the structural
evaluation and traffic volume. An earlier version of this app also handed those ratings
to the model as features, and the decision tree scored **0.995 accuracy** — it was
re-deriving its own target rather than predicting anything. The rule term
`structural_evaluation <= 4` alone recovers 67.5% of all positive labels.

Four features are therefore excluded from the model: `structural_evaluation`, `deck`,
`superstructure`, `substructure`. They remain visible in the Bridge Explorer as context
for a human reviewer.

Holdout performance without them, on a class-balanced 60,000-bridge sample:

| model | accuracy | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| Decision Tree | 0.816 | 0.866 | 0.747 | 0.802 |
| Random Forest | 0.831 | 0.829 | 0.833 | 0.831 |

These are the figures the deployed app reports; expect third-decimal drift across
scikit-learn versions.

## Inputs

The models use bridge inventory attributes that are independent of the label rule:

- state and bridge identifier
- bridge age and structure dimensions
- average daily traffic and truck traffic
- operating rating and designated inspection frequency
- design load, material/design category, and service type
- scour criticality

Condition ratings and structural evaluation are loaded and displayed, but deliberately
withheld from the models — see the leakage audit above.

## Screenshots

### Dashboard Overview

![Dashboard overview](assets/screenshots/overview.png)

### State Overview

![State overview](assets/screenshots/state-overview.png)

### Model Lab

![Model lab](assets/screenshots/model-lab.png)

### Feature Importance

![Random Forest feature importance](assets/screenshots/feature-importance.png)

With the four rating features removed, importance falls on `operating_rating`,
`bridge_age`, `design_load` and `designated_inspection_frequency` — load capacity,
age and inspection cadence rather than a restatement of the label. Selecting the
Decision Tree in the same tab swaps this chart for the printed rule path; the demo
recording above walks through it.

### Bridge Explorer

![Bridge explorer](assets/screenshots/bridge-explorer.png)

## Data Source

BridgeWatch uses the FHWA National Bridge Inventory. A dtype-optimised Parquet snapshot of
all 469,434 bridges across 53 state codes is committed at
`data/processed/bridgewatch_2025.parquet` (12.6 MB), so the app loads in ~0.2s instead of
downloading and fixed-width-parsing the 53 MB source file at boot.

To refresh when FHWA publishes a new year:

```bash
python -m src.data
```

Reference pages:

- [FHWA National Bridge Inventory downloads](https://www.fhwa.dot.gov/bridge/nbi/ascii2025.cfm)
- [FHWA NBI record format](https://www.fhwa.dot.gov/bridge/nbi/format.cfm)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The committed Parquet snapshot means no download is needed. Peak memory is ~701 MB, inside
Streamlit Community Cloud's 1 GB ceiling; cold start is ~13s. The hosted build at
[rei-bridgewatch.streamlit.app](https://rei-bridgewatch.streamlit.app/) runs from this same
branch.

## Repo Structure

- `app.py` - Streamlit dashboard interface
- `src/constants.py` - data paths, FHWA source configuration, and field metadata
- `src/data.py` - download, parsing, cleaning, and feature engineering pipeline
- `src/modeling.py` - model training, metrics, feature importance, and prediction helpers
- `src/visuals.py` - charts for confusion matrices, state summaries, and feature importance
- `assets/media/` - demo GIF and full recording
- `assets/screenshots/` - README screenshots

## GitHub Description

Interactive dashboard for explainable bridge inspection prioritization using tree-based machine learning.
