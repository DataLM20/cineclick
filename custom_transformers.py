from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re

class FeatureWeightingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_weights=None):
        self.feature_weights = feature_weights if feature_weights is not None else {
            'overview': 1.0,
            'director': 3.0,
            'genres': 2.0,
            'actors': 1.5
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureWeightingTransformer attend un pandas.DataFrame en entrée.")

        weighted_texts = []
        for index, row in X.iterrows():
            combined_text = []
            for feature, weight in self.feature_weights.items():
                content = str(row.get(feature, '')).lower()
                content = re.sub(r'[^a-zA-Z0-9\s]', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()

                if content:
                    combined_text.extend([content] * int(weight))
            weighted_texts.append(' '.join(combined_text))

        return pd.Series(weighted_texts, index=X.index)