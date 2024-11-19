import numpy as np;
from sklearn.base import BaseEstimator, ClassifierMixin;
from sklearn.metrics import pairwise_distances;
from sklearn.utils import check_X_y, check_array;

class SemiBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_neighbors=5, n_estimators=100, max_iter=50, learning_rate=1.0, random_state=42):
        self.base_estimator = base_estimator;
        self.n_neighbors = n_neighbors;
        self.n_estimators = n_estimators;
        self.max_iter = max_iter;
        self.learning_rate = learning_rate;
        self.random_state = random_state;

    def train(self, X, y):
        X, y = check_X_y(X, y);
        
        self.classes_ = np.unique(y);
        self.X_ = X;
        self.y_ = y;

        # Initialize weights
        self.weights_ = np.ones(len(y)) / len(y);

        for iteration in range(self.max_iter):
            # Compute pairwise distances
            distances = pairwise_distances(X);
            np.fill_diagonal(distances, np.inf);
            np.nan_to_num(distances, posinf=1e10, neginf=-1e10, copy=False);

            # Find nearest neighbors
            neighbors = np.argsort(distances, axis=1)[:, :self.n_neighbors];

            # Compute similarity matrix
            S = np.exp(-distances ** 2 / (2. * np.var(distances)));

            # Update weights
            for i in range(len(y)):
                for j in neighbors[i]:
                    if y[i] == y[j]:
                        self.weights_[i] *= np.exp(-self.learning_rate * S[i, j]);
                    else:
                        self.weights_[i] *= np.exp(self.learning_rate * S[i, j]);

            # Normalize weights
            self.weights_ /= np.sum(self.weights_);
        
        return self;

    def predict(self, X):
        X = check_array(X);
        distances = pairwise_distances(X, self.X_);
        predictions = [];

        for i in range(len(X)):
            weighted_votes = np.zeros(len(self.classes_));
            for j in range(len(self.X_)):
                weighted_votes[self.y_[j]] += self.weights_[j] / distances[i, j];
            predictions.append(self.classes_[np.argmax(weighted_votes)]);

        return np.array(predictions);

    def predict_proba(self, X):
        X = check_array(X);
        distances = pairwise_distances(X, self.X_);
        proba = [];

        for i in range(len(X)):
            weighted_votes = np.zeros(len(self.classes_));
            for j in range(len(self.X_)):
                weighted_votes[self.y_[j]] += self.weights_[j] / distances[i, j];
            proba.append(weighted_votes / np.sum(weighted_votes));

        return np.array(proba);