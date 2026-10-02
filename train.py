import numpy as np
import matplotlib.pyplot as plt

from model import LinearRegressionScratch


# ============================================================
# 1. CREATE DATASET
# ============================================================

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)


# ============================================================
# 2. CREATE MODEL
# ============================================================

model = LinearRegressionScratch(
    learning_rate=0.01,
    epochs=1000
)


# ============================================================
# 3. TRAIN MODEL
# ============================================================

model.fit(X, y)


# ============================================================
# 4. MAKE PREDICTIONS
# ============================================================

predictions = model.predict(X)


# ============================================================
# 5. DISPLAY MODEL PARAMETERS
# ============================================================

print("=" * 50)
print("LINEAR REGRESSION FROM SCRATCH")
print("=" * 50)

print(f"Weight (w): {model.w:.4f}")
print(f"Bias (b):   {model.b:.4f}")


# ============================================================
# 6. DISPLAY PREDICTIONS
# ============================================================

print("\nPredictions:")

for actual, predicted in zip(y, predictions):
    print(f"Actual: {actual:.2f} | Predicted: {predicted:.2f}")


# ============================================================
# 7. CALCULATE METRICS
# ============================================================

errors = y - predictions

mse = np.mean(errors ** 2)

mae = np.mean(np.abs(errors))

rmse = np.sqrt(mse)

ss_total = np.sum((y - np.mean(y)) ** 2)

ss_residual = np.sum((y - predictions) ** 2)

r2 = 1 - (ss_residual / ss_total)


print("\nEvaluation Metrics:")
print(f"MSE:  {mse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R²:   {r2:.6f}")


# ============================================================
# 8. PREDICT NEW / UNSEEN DATA
# ============================================================

X_new = np.array([6, 7, 8], dtype=float)

new_predictions = model.predict(X_new)

print("\nPredictions for unseen data:")

for x, prediction in zip(X_new, new_predictions):
    print(f"X = {x:.0f} -> Prediction = {prediction:.2f}")


# ============================================================
# 9. VISUALIZE REGRESSION LINE
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(X, y, label="Actual Data")

plt.plot(
    X,
    predictions,
    label="Regression Line"
)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression From Scratch")

plt.legend()
plt.grid(True)
plt.savefig("images/regression-line.png")
plt.show()



# ============================================================
# 10. VISUALIZE LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(model.loss_history)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss vs Epochs")

plt.grid(True)
plt.savefig("images/loss-curve.png")
plt.show()

