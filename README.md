# Loan Approval Prediction System
**An explainable AI system that predicts loan approvals and tells you exactly why — not just yes or no.**

## What This Project Does

Banks approve or reject loans every day without explaining their reasoning. This project builds a machine learning system that fixes that problem.

You give it an applicant's details. It tells you whether the loan should be approved, how confident it is, and — most importantly — **the exact reasons behind the decision** in plain English.



## Live Example Output

Applicant: Married graduate, good credit history, urban property, stable income

Decision        →  APPROVED ✅
Confidence      →  82%
Risk Level      →  LOW

Top Reasons:
  ✅ Good credit history        (strongest positive signal)
  ✅ Urban property location    (higher approval zone)
  ✅ Healthy income after EMI   (repayment capacity confirmed)
  ⚠️ High loan amount requested (slight negative pull)

Advice: Strong application. Maintain credit history before submission.


## Dataset Overview

![Target Distribution](images/target_distribution.png)

**614 real loan applicants** from the Kaggle Loan Prediction dataset.
The dataset has a real-world class imbalance — **68.7% Approved vs 31.3% Rejected** — which was fixed using SMOTE before training.



## Results

Six machine learning models were trained and compared. The best model — a **tuned Random Forest** — achieved:

| Accuracy | 82.93% |
| Precision | 84.04% |
| Recall | 92.94% |
| F1-Score | 88.27% |
| ROC-AUC | 82.17% |
| Cross-Validation AUC | **89.54%** |

> **89.54% cross-validation AUC** means the model generalises well to unseen applicants — not just the ones it trained on.



## Model Comparison

![Model Comparison](images/model_comparison.png)

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 82.93% | 80.34% |
| Decision Tree | 82.11% | 80.62% |
| **Random Forest** *(selected)* | **82.93%** | **82.17%** |
| SVM | 85.37% | 72.38% |
| Naive Bayes | 85.37% | 78.20% |
| XGBoost | 76.42% | 77.40% |

Random Forest was selected as the final model — best balance of accuracy and generalisation, and supports SHAP explainability natively.



## ROC Curves — All Models

![ROC Curves](images/roc_curves.png)

All six models plotted on a single chart. The closer a curve hugs the top-left corner, the better the model. The dashed diagonal line represents a model that guesses randomly.



## Feature Correlation Heatmap

![Correlation Heatmap](images/correlation_heatmap.png)

**Credit History** has the strongest positive correlation with Loan Status — applicants with good credit history had a **79.6% approval rate** vs only **7.9%** for those with bad credit history.



## Confusion Matrices

![Confusion Matrices](images/confusion_matrices.png)

Shows exactly where each model makes mistakes — False Positives (approved but shouldn't be) and False Negatives (rejected but should have been approved).



## Methodology


Applicant details entered
        ↓
Data cleaned + key features engineered
(Total Income · Monthly EMI · Balance after EMI)
        ↓
6 ML models trained and compared
        ↓
Best model selected and fine-tuned via GridSearchCV
(Best CV ROC-AUC: 89.54%)
        ↓
SHAP explains every individual decision
        ↓
Full report: Decision + Confidence + Reasons + Advice

## Key Technical Decisions

**Why SMOTE?**
The data had more approvals than rejections (2.2:1 ratio). Without fixing this, the model would learn to always predict approval and still look 68% accurate. SMOTE created synthetic rejection examples to balance the training set — applied only on training data, never test data.

**Why log transform income?**
One applicant earned ₹81,000/month while most earned under ₹6,000. That outlier would distort the model. Log transform compresses the scale so all income ranges are treated fairly.

**Why ROC-AUC over accuracy?**
A model that always says "Approved" gets 68.7% accuracy without learning anything useful. ROC-AUC measures whether the model can actually distinguish approvals from rejections — a far more honest metric.

**Why SHAP?**
SHAP (SHapley Additive exPlanations) comes from game theory. It calculates exactly how much each feature pushed a prediction toward approval or rejection — for every single applicant individually, not just on average.



## Tech Stack


| Python | Core language |
| pandas | Data cleaning and manipulation |
| scikit-learn | ML models and GridSearchCV tuning |
| XGBoost | Gradient boosting model |
| SHAP | Per-applicant explainability |
| imbalanced-learn | SMOTE for class balancing |
| matplotlib / seaborn | Visualisations |
| pickle | Save and load trained model |
