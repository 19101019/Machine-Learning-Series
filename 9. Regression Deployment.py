

# You're using accuracy_score, which only applies to classification models, but your current model (XGBRegressor) is a regression model.
# For regression, you should use metrics like R², MAE, or RMSE, not accuracy.

# --- 6. EVALUATE THE MODEL ---
# Make predictions on the unseen (scaled) test data
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

# R-squared tells us that our model explains about 58% of the variability
# in the house prices. It's a decent start but could be improved!
