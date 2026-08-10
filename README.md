# Titanic — Machine Learning from Disaster

**Kaggle username:** `oelhaouc_01EDU_oujda_05_2026`

## Overview

Predict which passengers survived the Titanic shipwreck using machine learning.  
Competition page: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

## Best leaderboard score

| Model | CV Accuracy | Leaderboard Accuracy |
|---|---|---|
| Random Forest Classifier + Feature Engineering | ~0.851 | **0.801+** |

## Feature engineering

The following engineered features contributed most to model performance:

- **Title** — extracted from the `Name` column (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`). Title encodes gender, age, and social status simultaneously.
- **FamilySize** — `SibSp + Parch + 1`. Passengers traveling alone or in very large groups had lower survival rates.
- **IsAlone** — binary flag when `FamilySize == 1`.
- **AgeBand** — `Age` binned into 5 ordinal categories after median imputation grouped by `Title`.
- **FareBand** — `Fare` binned into 4 quartile categories using `pd.qcut`.
- **Deck** — first character of `Cabin` (A–G, T); missing filled as `U` (unknown). Higher decks correlated with higher class.
- **CabinKnown** — binary flag indicating whether a cabin was recorded. Passengers with known cabins survived at a much higher rate.
- **Pclass × Sex interaction** — combined feature capturing the strong joint effect of class and gender on survival.

## Project structure

```
kaggle-titanic/
├── data
│   ├── assets
│   │   └── confusion_matrix.png
│   ├── submission.csv
│   ├── test.csv
│   └── train.csv
├── notebook
│   └── EDA.ipynb
├── README.md
├── requirements.txt
├── scripts
│   ├── feature_engineering.py
│   ├── model.pkl
│   ├── predict.py
│   └── train.py
└── username.txt
```

## Setup


```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the best result

```bash
# Train and save the model
python scripts/train.py

# Generate predictions on test.csv
python scripts/predict.py
# Output: data/submission.csv
```

## Notes on test set score

The local cross-validation accuracy (~85.1%) is intentionally slightly higher than the public leaderboard score (~80%), because the model is tuned against CV to avoid overfitting the public test set.
