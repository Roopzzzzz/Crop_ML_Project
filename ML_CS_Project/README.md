# Automated Irrigation and Crop Health Monitoring System Using Machine Learning

Crop recommendation and soil-climate zone analysis using machine learning.

**Dataset:** Crop Recommendation Dataset (Kaggle) - 2,200 rows x 8 columns, 22 crops
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

## What is inside
| File | Purpose |
|---|---|
| `Crop_ML_Project.ipynb` | Full workflow: EDA, classification, prediction, clustering, model saving, reuse |
| `saved_models/` | Serialised models (`.joblib`, `.pkl`) |
| `predict.py` | Reuse the saved model from the command line, no retraining |
| `requirements.txt` | Python packages needed |

## ML tasks
- **Classification** - 6 algorithms compared (Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Naive Bayes)
- **Prediction** - recommends a crop for new, unseen soil/climate readings
- **Clustering** - K-Means (k=4) groups records into soil-climate zones

## Saved model files
| File | Contents |
|---|---|
| `crop_classifier.joblib` / `.pkl` | Best trained classifier |
| `label_encoder.joblib` | Converts predicted numbers back to crop names |
| `feature_columns.joblib` | Input feature order |
| `scaler.joblib`, `kmeans_model.joblib` | Clustering (zone) models |

## Reuse the model
```bash
pip install -r requirements.txt
python predict.py 90 42 43 20.8 82 6.5 202.9
```
or in Python:
```python
import joblib, pandas as pd
model = joblib.load("saved_models/crop_classifier.joblib")
le    = joblib.load("saved_models/label_encoder.joblib")
cols  = joblib.load("saved_models/feature_columns.joblib")
sample = pd.DataFrame([[90, 42, 43, 20.8, 82.0, 6.5, 202.9]], columns=cols)
print(le.inverse_transform(model.predict(sample))[0])
```
Note: use the same scikit-learn version the model was trained with (see `requirements.txt`).
