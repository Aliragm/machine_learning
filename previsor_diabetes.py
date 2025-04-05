#feito no colab

import pandas as pd
import sklearn as skl

df = pd.read_csv('diabetes.csv')

x = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = skl.preprocessing.StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled, columns=x.columns)

df_full = pd.concat([x_scaled, y], axis=1)
X_train, X_test, y_train_binary, y_test_binary = skl.model_selection.train_test_split(x_scaled, y, test_size=0.2, random_state=42)

log_reg = skl.linear_model.LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train_binary)

y_pred_binary = log_reg.predict(X_test)

accuracy = skl.metrics.accuracy_score(y_test_binary, y_pred_binary)
paccuracy = skl.metrics.accuracy_score(y_test_binary, y_pred_binary)
conf_matrix = skl.metrics.confusion_matrix(y_test_binary, y_pred_binary)
class_report = skl.metrics.classification_report(y_test_binary, y_pred_binary)

print("""
============================================
         Model Performance Metrics
============================================
""")
print(f"1. Accuracy: {accuracy:.4f}")

print("\n2. Confusion Matrix:")
print(f"""
          | Predicted 0 | Predicted 1 |
----------|-------------|-------------|
Actual 0  | {conf_matrix[0,0]:^11} | {conf_matrix[0,1]:^11} |
Actual 1  | {conf_matrix[1,0]:^11} | {conf_matrix[1,1]:^11} |
""")

print("3. Classification Report:")
print(class_report)

print("4. Additional Metrics:")
print(f"- Precision: {skl.metrics.precision_score(y_test_binary, y_pred_binary):.4f}")
print(f"- Recall: {skl.metrics.recall_score(y_test_binary, y_pred_binary):.4f}")
print(f"- F1-Score: {skl.metrics.f1_score(y_test_binary, y_pred_binary):.4f}")
print(f"- AUC-ROC: {skl.metrics.roc_auc_score(y_test_binary, y_pred_binary):.4f}")

