<div align="center">

# Comment Category Prediction

### Multi-modal NLP classification pipeline — predicting comment categories from text, engagement, and identity features.

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-2980B9?style=flat-square&logo=lightgbm&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

</div>

---

## 🧠 Problem Statement

Given a dataset of online comments with metadata (engagement, identity signals, platform flags), predict which of **4 categories** each comment belongs to:

| Label | Category |
|---|---|
| 0 | General / Neutral conversation |
| 1 | Identity-based content (race, religion, gender) |
| 2 | Political / Systemic criticism |
| 3 | Aggressive / Toxic (short, direct) |

**Evaluation metric:** Macro F1-score (penalizes poor performance on minority classes)

---

## 📊 Dataset & Key Findings

**Class distribution (severe imbalance):**
- Class 0: ~58% (majority)
- Class 1 & 2: moderately represented
- Class 3: ~2% (extreme minority)

**EDA insights that drove feature engineering:**

- Class 3 comments are significantly shorter (median ~128 chars vs ~199–246 for others)
- Class 1 has strong correlation with race/religion/gender identity fields
- Class 2 shows political vocabulary patterns
- `if_1` and `if_2` (internal platform flags) showed strong class separation → treated as categorical, not numeric
- Upvote/downvote distributions are heavily right-skewed → log-transformed

---

## 🛠️ Feature Engineering

```
Text Features
├── Word-level TF-IDF  (unigrams + bigrams, 50k features)
└── Character-level TF-IDF  (3–5 char n-grams, 100k features)
    └── Combined via FeatureUnion

Numeric Features
├── word_count
├── log_upvote  = log1p(upvote)
└── log_downvote = log1p(downvote)

Categorical Features
├── race, religion, gender, disability
└── if_1, if_2  →  OneHotEncoded
```

All features unified in a single `ColumnTransformer` pipeline.

---

## ⚙️ Model Pipeline

```python
final_model = Pipeline([
    ('preprocessor', ColumnTransformer([
        ("text", FeatureUnion([word_tfidf, char_tfidf]), "comment"),
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(), cat_cols)
    ])),
    ('model', SGDClassifier(
        loss='log_loss',
        penalty='elasticnet',
        alpha=5e-06,
        l1_ratio=0.15,
        class_weight={0:1, 1:1.5, 2:1, 3:2.5},  # custom weights for imbalance
        max_iter=5000,
        random_state=42
    ))
])
```

**Models compared:**
| Model | Notes |
|---|---|
| Logistic Regression | Baseline |
| SGDClassifier ✅ | **Final model** — fastest, best F1 after tuning |
| LinearSVC | Comparable but slower |
| LightGBM | Tested, lower F1 on this text-heavy task |

---

## 🔧 Handling Class Imbalance

Class 3 (~2%) was the hardest to classify. Two strategies used:

1. **Custom class weights** — `{0:1, 1:1.5, 2:1, 3:2.5}` penalizes misclassification of rare classes more heavily
2. **Macro F1 optimization** — tuned with `scoring="f1_macro"` in RandomizedSearchCV to prevent the model from ignoring minority classes

---

## 📁 Repository Structure

```
comment-category-prediction/
├── notebook.ipynb          # Full pipeline: EDA → features → model → submission
├── submission.csv          # Kaggle submission file
└── README.md
```

> **Note:** Dataset not included (Kaggle competition data). Download from the competition page and place in `/data/`.

---

## 🚀 Run Locally

```bash
git clone https://github.com/moonejs/comment-category-prediction.git
cd comment-category-prediction

pip install numpy pandas scikit-learn lightgbm matplotlib seaborn

# Place train.csv, test.csv, Sample.csv in the working directory
jupyter notebook notebook.ipynb
```

---

## 👤 Author

**Litesh** — IIT Madras, BS in Data Science & Applications

[![GitHub](https://img.shields.io/badge/@moonejs-181717?style=flat-square&logo=github)](https://github.com/moonejs)
[![Email](https://img.shields.io/badge/24f2003468@ds.study.iitm.ac.in-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:24f2003468@ds.study.iitm.ac.in)

---

<div align="center">
<sub>IIT Madras · Machine Learning Project Course · Kaggle Competition</sub>
</div>
