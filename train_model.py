from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "matches.csv"
MODEL_PATH = BASE_DIR / "ipl_model.joblib"
FEATURES = ["team", "opponent", "venue", "won_toss", "toss_decision"]

matches = pd.read_csv(DATA_PATH).dropna(subset=["team_1", "team_2", "winner", "venue", "toss_winner", "toss_decision"])
matches = matches[matches.apply(lambda r: r["winner"] in (r["team_1"], r["team_2"]), axis=1)].copy()
rows = []
for _, m in matches.iterrows():
    for team, opponent in [(m["team_1"], m["team_2"]), (m["team_2"], m["team_1"])]:
        rows.append({"team": str(team), "opponent": str(opponent), "venue": str(m["venue"]),
                     "won_toss": int(str(m["toss_winner"]) == str(team)),
                     "toss_decision": str(m["toss_decision"]).lower(),
                     "won": int(str(m["winner"]) == str(team))})
data = pd.DataFrame(rows)
X, y = data[FEATURES], data["won"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
preprocessor = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), FEATURES)])
model = Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42))])
model.fit(X_train, y_train)
print(f"Held-out test accuracy: {accuracy_score(y_test, model.predict(X_test)):.1%}")
print(classification_report(y_test, model.predict(X_test), zero_division=0))
model.fit(X, y)
joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
