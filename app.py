from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "matches.csv"
FEATURES = ["team", "opponent", "venue", "won_toss", "toss_decision"]

st.set_page_config(page_title="IPL Winning Prediction", page_icon="🏏", layout="centered")
st.title("🏏 IPL Winning Prediction")
st.write("A small machine-learning demo that estimates which of two IPL teams is more likely to win.")
st.caption("Educational prototype only — predictions are not guaranteed and the supplied dataset covers just one season.")

@st.cache_data
def load_matches():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = ["team_1", "team_2", "winner", "venue", "toss_winner", "toss_decision"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    df = df.dropna(subset=required).copy()
    # Keep only decisive results where the winner is one of the two participating teams.
    df = df[df.apply(lambda r: r["winner"] in (r["team_1"], r["team_2"]), axis=1)].copy()
    return df

def make_team_rows(matches):
    rows = []
    for _, m in matches.iterrows():
        for team, opponent in [(m["team_1"], m["team_2"]), (m["team_2"], m["team_1"])]:
            rows.append({
                "team": str(team),
                "opponent": str(opponent),
                "venue": str(m["venue"]),
                "won_toss": int(str(m["toss_winner"]) == str(team)),
                "toss_decision": str(m["toss_decision"]).lower(),
                "won": int(str(m["winner"]) == str(team)),
            })
    return pd.DataFrame(rows)

@st.cache_resource
def train_model():
    matches = load_matches()
    rows = make_team_rows(matches)
    if len(rows) < 20 or rows["won"].nunique() < 2:
        raise ValueError("Not enough valid match records to train the model.")
    X = rows[FEATURES]
    y = rows["won"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    prep = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), FEATURES)])
    model = Pipeline([
        ("preprocessor", prep),
        ("classifier", RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
                                               class_weight="balanced", random_state=42))
    ])
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    # Refit on all available examples after obtaining a held-out evaluation estimate.
    model.fit(X, y)
    teams = sorted(set(matches["team_1"].astype(str)) | set(matches["team_2"].astype(str)))
    venues = sorted(matches["venue"].astype(str).unique())
    return model, accuracy, teams, venues, len(matches)

try:
    model, accuracy, teams, venues, match_count = train_model()
    st.success(f"Model trained on {match_count} completed matches.")
    with st.expander("About this model"):
        st.write(f"Held-out test accuracy on team-perspective rows: **{accuracy:.1%}**")
        st.write("This is a small random train/test split, not proof of real-world predictive accuracy. The dataset has limited coverage.")
    st.subheader("Enter match details")
    c1, c2 = st.columns(2)
    with c1:
        team1 = st.selectbox("Team 1", teams, index=0)
    with c2:
        opponent_options = [t for t in teams if t != team1]
        team2 = st.selectbox("Team 2", opponent_options, index=0)
    venue = st.selectbox("Venue", venues)
    toss_winner = st.selectbox("Toss winner", [team1, team2])
    toss_decision = st.radio("Toss decision", ["bat", "field"], horizontal=True)

    if st.button("Predict winner", type="primary", use_container_width=True):
        records = pd.DataFrame([
            {"team": team1, "opponent": team2, "venue": venue,
             "won_toss": int(toss_winner == team1), "toss_decision": toss_decision},
            {"team": team2, "opponent": team1, "venue": venue,
             "won_toss": int(toss_winner == team2), "toss_decision": toss_decision},
        ])
        probs = model.predict_proba(records)[:, list(model.named_steps["classifier"].classes_).index(1)]
        total = float(probs.sum())
        p1, p2 = (0.5, 0.5) if total == 0 else (float(probs[0] / total), float(probs[1] / total))
        winner = team1 if p1 >= p2 else team2
        st.subheader("Prediction")
        st.success(f"Predicted winner: {winner}")
        st.write(f"**{team1}:** {p1:.1%} estimated relative chance")
        st.progress(p1)
        st.write(f"**{team2}:** {p2:.1%} estimated relative chance")
        st.progress(p2)
        st.caption("These are normalized model scores for the two selected teams, not calibrated betting odds or a guarantee.")
except Exception as exc:
    st.error(f"Could not start the prediction app: {exc}")
    st.info("Check that dataset/matches.csv exists and install dependencies from requirements.txt.")
