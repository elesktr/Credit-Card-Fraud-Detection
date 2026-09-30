#comparing two variations of the rfc variable to point out the balance difference

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef 
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("creditcard.csv")

X = data.drop(['Class'], axis = 1)
Y = data["Class"]

xTrain, xTest, yTrain, yTest = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

#baseline rfc

baseline_rfc = RandomForestClassifier(
    random_state=42
)

baseline_rfc.fit(xTrain, yTrain)
yPred_baseline = baseline_rfc.predict(xTest)

baseline_metrics = {
    "Accuracy": accuracy_score(yTest, yPred_baseline),
    "Precision": precision_score(yTest, yPred_baseline),
    "Recall": recall_score(yTest, yPred_baseline),
    "F1-Score": f1_score(yTest, yPred_baseline),
    "MCC": matthews_corrcoef(yTest, yPred_baseline)
}

#class-weighted rfc

weighted_rfc = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

weighted_rfc.fit(xTrain, yTrain)
yPred_weighted = weighted_rfc.predict(xTest)

weighted_metrics = {
    "Accuracy": accuracy_score(yTest, yPred_weighted),
    "Precision": precision_score(yTest, yPred_weighted),
    "Recall": recall_score(yTest, yPred_weighted),
    "F1-Score": f1_score(yTest, yPred_weighted),
    "MCC": matthews_corrcoef(yTest, yPred_weighted)
}


#comp rfc

comparison = pd.DataFrame(
    [baseline_metrics, weighted_metrics],
    index=["Baseline Random Forest", "Class-Weighted Random Forest"]
)

print("\nModel Comparison:")
print(comparison.round(4))

different_predictions = np.sum(
    yPred_baseline != yPred_weighted
)

print("\nPrediction Comparison:")
print(f"Number of different predictions: {different_predictions}")