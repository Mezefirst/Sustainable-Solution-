import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("materials.csv")
X = df.drop("name", axis=1)
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_scaled, df["name"])

import joblib
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
