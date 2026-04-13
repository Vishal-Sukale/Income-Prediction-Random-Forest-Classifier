# Income Prediction Using Random Forest Classifier with Hyperparameter Tuning

## Problem Statement
The income classification problem involves predicting whether an individual
earns more or less than $50K annually based on demographic and employment
attributes such as age, education, occupation, marital status, and work hours.

A single Decision Tree often overfits and lacks generalization. This project
addresses that limitation by using Random Forest — a powerful ensemble method
that combines multiple Decision Trees to deliver more accurate and stable
predictions on the Adult Census Dataset (32,561 records).

## Objective
- Predict income category (<=50K or >50K) using Random Forest Classifier
- Handle missing values encoded as ' ?' in the Adult dataset
- Apply One-Hot Encoding with drop_first=True to avoid multicollinearity
- Improve model performance using RandomizedSearchCV Hyperparameter Tuning
- Evaluate model using Accuracy, Precision, Recall and F1 Score
- Use Classification Report for detailed class-wise performance analysis
- Compare Before and After tuning performance

## Dataset
| Detail | Info |
|---|---|
| Name | Adult Census Income Dataset |
| Records | 32,561 |
| Features | 14 |
| Target | income (<=50K or >50K) |
| Missing Values | Encoded as ' ?' |

## Tech Stack
| Tool | Usage |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Operations |
| Scikit-learn | ML Model & Evaluation |
| Jupyter Notebook | Development Environment |

## ML Pipeline
```
1. Import Libraries
2. Load Dataset
3. Data Understanding
4. Data Preprocessing
   - Handle Missing Values (' ?' → NaN → dropna)
   - One-Hot Encoding (get_dummies, drop_first=True)
5. Model Building
   - Split Features & Target
   - Train-Test Split (80/20)
   - Model Before Tuning
   - RandomizedSearchCV Hyperparameter Tuning
   - Model After Tuning
6. Evaluation & Comparison
```

## Results

| Metric | Before Tuning | After Tuning |
|---|---|---|
| **Accuracy** | 85.18% | **86.09%**  |
| **Precision** | 0.718 | **0.771**  |
| **Recall** | 0.612 | 0.584 |
| **F1 Score** | 0.661 | **0.664** |

###  Best Parameters Found
| Parameter | Value |
|---|---|
| criterion | entropy |
| max_depth | 30 |
| n_estimators | 100 |
| min_samples_leaf | 2 |
| min_samples_split | 5 |

---

## Key Insights
- Accuracy improved from **85.18% → 86.09%** after tuning 
- Precision improved from **0.718 → 0.771** — fewer false positives 
- Recall slightly decreased from 0.612 → 0.584 due to stricter boundaries
- F1 Score improved from **0.661 → 0.664** — better overall balance 
- Random Forest outperformed single Decision Tree (~85% → 86.09%) 
- Class imbalance present: 4976 (<=50K) vs 1537 (>50K)
- RandomizedSearchCV with n_iter=20, cv=3 used for faster tuning
- Weighted avg F1 Score = 0.85 — strong model performance 

## Project Structure
```
Income-Prediction-Random-Forest-Classifier/
│
├── RF_Classifier_Project.ipynb   # Main Jupyter Notebook
├── Adult_Dataset.csv             # Dataset
└── README.md                     # Project Documentation
```
## How to Run

1. Clone the repository
```bash
git clone https://github.com/yourusername/Income-Prediction-Random-Forest-Classifier.git
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn
```

3. Open Jupyter Notebook
```bash
jupyter notebook RF_Classifier_Project.ipynb
```

## Conclusion
The Random Forest Classifier successfully predicted individual income
categories with an accuracy of **86.09%** after hyperparameter tuning —
an improvement from the **85.18%** baseline.

Compared to a single Decision Tree (~85%), Random Forest delivered
higher accuracy and precision by combining predictions from 100 trees,
reducing overfitting and improving generalization on unseen data.

RandomizedSearchCV efficiently found the best hyperparameters in
significantly less time compared to GridSearchCV, making it a practical
choice for large datasets. The Classification Report confirmed strong
performance with weighted average F1 Score of **0.85**.

This project demonstrates the real-world advantage of ensemble learning
over single models for income classification tasks involving class
imbalance and mixed data types.

##  Author
**Vishal Sukale**

