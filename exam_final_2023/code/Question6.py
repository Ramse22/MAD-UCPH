import random

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# a)
def read_data(file_name):
    data = pd.read_csv(file_name)
    data["Sex"] = np.where(data["Sex"] == "M", 0, 1)
    data["ASY"] = np.where(data["ChestPainType"] == "ASY", 1, 0)
    data["NAP"] = np.where(data["ChestPainType"] == "NAP", 1, 0)
    data["TA"] = np.where(data["ChestPainType"] == "TA", 1, 0)
    data["ATA"] = np.where(data["ChestPainType"] == "ATA", 1, 0)
    data = data.drop("ChestPainType", axis=1)
    data_features = data[
        ["Age", "RestingBP", "Cholesterol", "MaxHR", "Sex", "ASY", "NAP", "TA", "ATA"]
    ].values
    data_labels = data[["HeartDisease"]].values

    return data_features, data_labels


X_train, t_train = read_data("../data/heart_simplified_train_2023.csv")
X_val, t_val = read_data("../data/heart_simplified_validation_2023.csv")
X_test, t_test = read_data("../data/heart_simplified_validation_2023.csv")

# b)
RFC = RandomForestClassifier()
RFC.fit(X_train, t_train.ravel())
RFC.predict(X_train)
print(f" Accuracy for train set: {RFC.score(X_train, t_train)}")

# c) and d)
max_features = ["sqrt", "log2"]
criterion = ["gini", "entropy"]
max_depth = [2, 5, 7, 10, 15]


result_occ = {}
for _ in range(500):
    best_scores = (0, 0)
    best_params = None
    for _ in range(15):
        params = [
            random.choice(max_features),
            random.choice(criterion),
            random.choice(max_depth),
        ]
        RFC = RandomForestClassifier(
            max_features=params[0], criterion=params[1], max_depth=params[2]
        )
        RFC.fit(X_train, t_train.ravel())
        t_predict = RFC.predict(X_val)
        corrects = np.sum(t_val.ravel() == t_predict.ravel())
        probas = RFC.predict_proba(X_val)
        corrects_probas = []
        for idx, label in enumerate(t_predict.ravel()):
            if label == 1 and probas[idx, 1] > probas[idx, 0]:
                corrects_probas.append(probas[idx, 1])
            elif label == 0 and probas[idx, 0] < probas[idx, 1]:
                corrects_probas.append(probas[idx, 0])
        if best_scores[1] < np.mean(corrects_probas):
            best_scores = (corrects, np.mean(corrects_probas))
            best_params = params
            # print(
            #    f"criterion = {params[1]} ; max depth = {params[2]} ; max features = {params[0]} ; accuracy on validation data = {corrects/len(t_val)} ; number of correctly classified validation samples = {corrects}"
            # )
    print(f" best metrics: {best_scores}, best parameters: {best_params}")
    # add + 1 to occurrence
    result_occ[str(best_params)] = result_occ.get(str(best_params), 0) + 1

result_occ = {k: v for k, v in sorted(result_occ.items(), key=lambda item: item[1])}
print(result_occ)
plt.bar(range(len(result_occ)), list(result_occ.values()), align="center")
plt.xticks(range(len(result_occ)), list(result_occ.keys()), rotation=70)
plt.tight_layout()
plt.savefig("^parameters_occurrence.png")
