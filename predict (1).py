"""
predict.py - reuse the saved crop recommendation model WITHOUT retraining.

Usage:
    python predict.py                      # runs a demo with sample readings
    python predict.py 90 42 43 20.8 82 6.5 202.9
                       N  P  K  temp humidity ph rainfall
"""
import sys
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = "saved_models"

model   = joblib.load(f"{MODEL_DIR}/crop_classifier.joblib")
le      = joblib.load(f"{MODEL_DIR}/label_encoder.joblib")
columns = joblib.load(f"{MODEL_DIR}/feature_columns.joblib")
scaler  = joblib.load(f"{MODEL_DIR}/scaler.joblib")
kmeans  = joblib.load(f"{MODEL_DIR}/kmeans_model.joblib")


def recommend_crop(N, P, K, temperature, humidity, ph, rainfall, top_n=3):
    sample = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=columns)
    crop = le.inverse_transform(model.predict(sample))[0]
    result = {"recommended_crop": crop}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(sample)[0]
        top = np.argsort(proba)[::-1][:top_n]
        result["top_choices"] = [(le.classes_[i], round(float(proba[i]) * 100, 2)) for i in top]
    result["soil_climate_zone"] = int(kmeans.predict(scaler.transform(sample))[0])
    return result


if __name__ == "__main__":
    if len(sys.argv) == 8:
        vals = [float(v) for v in sys.argv[1:]]
        print(recommend_crop(*vals))
    else:
        demo = [
            (90, 42, 43, 20.8, 82.0, 6.5, 202.9),
            (20, 68, 20, 18.5, 16.0, 6.8, 78.0),
        ]
        for d in demo:
            print("Input :", d)
            print("Output:", recommend_crop(*d), "\n")
