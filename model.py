import numpy as np


class LinearRegressionScratch:

    def __init__(self, learning_rate=0.01, epochs=1000):
        """
        Initialize the Linear Regression model.

        Parameters:
        learning_rate : controls how large each gradient descent step is
        epochs        : number of times the model goes through the dataset
        """

        self.learning_rate = learning_rate
        self.epochs = epochs

        # Model parameters
        self.w = 0.0
        self.b = 0.0

        # Store loss after every epoch
        self.loss_history = []

    def predict(self, X):
        """
        Generate predictions using:

            y_hat = wx + b
        """

        return X * self.w + self.b

    def fit(self, X, y):
        """
        Train the model using gradient descent.
        """

        # Reset parameters
        self.w = 0.0
        self.b = 0.0
        self.loss_history = []

        n = len(X)

        for epoch in range(self.epochs):

            # --------------------------------
            # 1. Make predictions
            # --------------------------------
            predictions = self.predict(X)

            # --------------------------------
            # 2. Calculate error
            # --------------------------------
            error = predictions - y

            # --------------------------------
            # 3. Calculate MSE
            # --------------------------------
            loss = np.mean(error ** 2)

            # --------------------------------
            # 4. Calculate gradients
            # --------------------------------

            # Gradient with respect to weight
            dw = (2 / n) * np.sum(X * error)

            # Gradient with respect to bias
            db = (2 / n) * np.sum(error)

            # --------------------------------
            # 5. Update parameters
            # --------------------------------
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            # --------------------------------
            # 6. Store loss
            # --------------------------------
            self.loss_history.append(loss)

        return self