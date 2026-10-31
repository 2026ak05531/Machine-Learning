# NPHA Doctor Visits – Multi-class Classification

## Problem statement
Implement five classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) on the same dataset, evaluate them with Accuracy, AUC, Precision, Recall, F1 and MCC, and deploy a Streamlit app to explore them.
Here the task is to predict how many times an older adult visits a doctor per year (`Number of Doctors Visited`: 1 = 0-1 visits, 2 = 2-3 visits, 3 = 4+ visits) from health, sleep and demographic survey answers.

## Dataset description
- **Source:** National Poll on Healthy Aging (NPHA) – doctor visits dataset (`NPHA-doctor-visits.csv`, UCI/Kaggle).
- **Size:** 714 instances, 14 input features + 1 target (3 classes: 131 / 372 / 211).
- **Features:** physical, mental and dental health ratings; employment; five "keeps patient from sleeping" indicators (stress, medication, pain, bathroom, unknown); trouble sleeping; prescription sleep medication; race; gender. All are integer-coded categorical/ordinal values.
- **Preprocessing:** `Age` dropped (constant, all = 2); `-1` ("refused") treated as missing and imputed with the column mode; features standardised for Logistic Regression, KNN and SVM; stratified 80/20 train/test split (`random_state=42`). The test split is saved as `test_data.csv`.
- **Metric averaging:** Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest macro-averaged.

## GitHub repository link
<ADD YOUR GITHUB REPO LINK HERE>

## Models used
| **ML Model Name** | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1** | **MCC** |
| --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 0.5524 | 0.6305 | 0.3872 | 0.3790 | 0.3279 | 0.1547 |
| **Decision Tree** | 0.4965 | 0.5809 | 0.3717 | 0.3260 | 0.2616 | -0.0187 |
| **KNN** | 0.4755 | 0.5189 | 0.3664 | 0.3504 | 0.3362 | 0.0376 |
| **Naive Bayes** | 0.2937 | 0.6264 | 0.4333 | 0.4282 | 0.2973 | 0.1427 |
| **Support Vector Machine** | 0.5455 | 0.4874 | 0.3636 | 0.3676 | 0.3074 | 0.1321 |

## Observations
| **ML Model Name** | **Observation about model performance** |
| --- | --- |
| **Logistic Regression** | Best accuracy (0.552) and best AUC (0.631) and MCC (0.155). Its probabilities rank classes better than chance, so a mostly linear signal exists, but it leans on the majority class (2-3 visits), keeping macro recall near 0.38. |
| **Decision Tree** | Accuracy 0.497 with MCC close to zero (-0.019), meaning it is barely better than guessing. Even with depth limited to 5 it fits noise in the small training set; the weakest overall. |
| **KNN** | Lowest accuracy among the non-NB models (0.476) and AUC near 0.52. Features are coarse integer codes, so many points are tied or equidistant and neighbours are not informative. Its macro F1 (0.336) is slightly the highest only because it predicts minority classes more often. |
| **Naive Bayes** | Accuracy is very low (0.294) because Gaussian assumptions fit coded categorical features poorly and it over-predicts minority classes. However it has the highest macro precision (0.433) and recall (0.428) and second-best AUC (0.626), so it is the most balanced across classes. |
| **Support Vector Machine** | Accuracy close to Logistic Regression (0.546), but AUC (0.487) is below chance, indicating its probability estimates (Platt scaling) are poorly calibrated and the RBF kernel is not separating the classes; it mostly predicts the majority class. |
| **Overall, Winner for your dataset?** | **Logistic Regression** – highest accuracy, AUC and MCC, with competitive F1. Performance of all models is modest (the target is hard to predict from these survey features and the data is small and imbalanced), so improvements would likely need class weighting, tuning and more data. |

## Repository structure
```
app.py              Streamlit app (upload CSV, pick a model, see metrics + confusion matrix)
train_models.py     Preprocessing, training, evaluation, model saving
requirements.txt
README.md
test_data.csv       20% hold-out test split (with target column)
NPHA-doctor-visits.csv
model/              logistic_regression, decision_tree, knn, naive_bayes, support_vector_machine (.joblib), preprocess.joblib, metrics.csv
```

## Run locally
```
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
