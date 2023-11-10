from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

#bayesian classifier of lab 1 

class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation of a Gaussian Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance

    def fit(self, X, y):
        """
        Fit the classifier. Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        Parameters:
        - X: array-like or pd.DataFrame, shape (n_samples, n_features)
            The training input samples.
        - y: array-like or pd.Series, shape (n_samples,)
            The target values.

        Returns:
        - self: object
            Returns self.
        """
        self.classes_ = np.unique(y)
        self.X_mean_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])

        if self.use_unit_variance:
            self.X_var_ = np.ones_like(self.X_mean_)
        else:
            self.X_var_ = np.array([X[y == c].var(axis=0) for c in self.classes_])

        return self

    def _calculate_likelihood(self, X, class_idx):
        """Calculate the likelihood of each feature for a given class."""
        mean = self.X_mean_[class_idx]
        var = self.X_var_[class_idx]
        exponent = -0.5 * np.square((X - mean) / var)
        return np.exp(exponent) / (np.sqrt(2 * np.pi * var))

    def predict(self, X):
        """
        Make predictions for X based on the Gaussian Naive Bayes model.

        Parameters:
        - X: array-like or pd.DataFrame, shape (n_samples, n_features)
            The input samples.

        Returns:
        - predictions: array, shape (n_samples,)
            Predicted class labels.
        """
        predictions = [np.argmax([np.sum(np.log(self._calculate_likelihood(x, c))) for c in range(len(self.classes_))]) for x in X]
        return np.array([self.classes_[p] for p in predictions])

    def score(self, X, y):
        """
        Return accuracy score on the predictions for X based on ground truth y.

        Parameters:
        - X: array-like or pd.DataFrame, shape (n_samples, n_features)
            The input samples.
        - y: array-like or pd.Series, shape (n_samples,)
            The true labels.

        Returns:
        - accuracy: float
            The accuracy of the model on the given data.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

#load data from csv