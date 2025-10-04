
# ---------------------------
# 13. Final Evaluation
# ---------------------------
y_test_pred = best_model.predict(X_test)

print("Final Accuracy:", accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))



# ---------------------------
# 14. Feature Importance
# ---------------------------
importances = best_model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.show()


