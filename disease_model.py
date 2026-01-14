import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

data = pd.read_csv("symptoms_dataset.csv")

X = data.drop("disease", axis=1)
y = data["disease"]

model = DecisionTreeClassifier()
model.fit(X, y)

pickle.dump(model, open("disease_model.pkl", "wb"))

print("Model trained successfully!")