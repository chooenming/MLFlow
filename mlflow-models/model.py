# this will be .pkl in mlflow artifacts

import mlflow.pyfunc

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer


class SentimentDetector(mlflow.pyfunc.PythonModel):
    def __init__(self) -> None:
        self.vectorizer = None
        self.model = None

    def load_context(self, context):
        """
        Getting context from MLFlow
        """
        import joblib
        # context in mlflow
        self.model = joblib.load(context.artifacts["model"]) # refers to artifacts in mlflow
        self.vectorizer = joblib.load(context.artifacts["vectorizer"])

    def predict(self, context, input_df: pd.DataFrame) -> np.ndarray:
        inputs = self.vectorizer.transform(input_df["review"])
        return self.model.predict(inputs)

