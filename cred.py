import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report


data = pd.read_csv("creditcard.csv")

print("Dataset shape:", data.shape)
print("\nMissing values:")
print(data.isnull().sum().sum())

print("\nClass distribution:")
print(data["Class"].value_counts())
print("\nClass distribution (%):")
print((data["Class"].value_counts(normalize=True) * 100).round(4))


fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))


corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


X = data.drop(['Class'], axis = 1)
Y = data["Class"]
print(X.shape)
print(Y.shape)

xData = X.values
yData = Y.values
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.2, random_state = 42)


rfc = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)
rfc.fit(xTrain, yTrain)

yPred = rfc.predict(xTest)

accuracy = accuracy_score(yTest, yPred)
precision = precision_score(yTest, yPred)
recall = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
mcc = matthews_corrcoef(yTest, yPred)

print("\nClassification Report:")
print(
    classification_report(
        yTest,
        yPred,
        target_names=["Normal", "Fraud"]
    )
)

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Prediction Class")
plt.ylabel("True Class")
plt.show()

feature_importance = pd.Series(
    rfc.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))

feature_importance.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 feature importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
