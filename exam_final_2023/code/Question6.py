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
print(X_test)
# b)


RFC = RandomForestClassifier()
predictor = RFC.fit(X_train, t_train.ravel())
print(predictor.score(X_test, t_test))

# c)
n_features = [np.sqrt(X_val.shape[1]), np.log2(X_val.shape[1])]
criterion = ["gini", "entropy"]
max_depth = [2, 5, 7, 10, 15]
