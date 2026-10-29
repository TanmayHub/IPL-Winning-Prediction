# IPL Winning Prediction (fixed project)

A beginner-friendly Streamlit + scikit-learn demonstration that estimates a winner between two selected IPL teams.

## Requirements
- Python 3.10–3.12 recommended
- Internet is needed only to install packages; the app itself reads the included CSV locally.

## Run on Windows
1. Extract this ZIP.
2. Open Command Prompt in the extracted `IPL_Winning_Prediction_Fixed` folder.
3. Create a virtual environment: `py -m venv .venv`
4. Activate it: `.venv\Scripts\activate`
5. Install packages: `python -m pip install -r requirements.txt`
6. Start the app: `python -m streamlit run app.py`
7. Open the local URL printed in the terminal (usually `http://localhost:8501`).

The app reads `dataset/matches.csv` using a path relative to `app.py`, so it does not depend on the current terminal folder.

## Optional: train and evaluate from the command line
Run `python train_model.py`. This prints a held-out test score and saves `ipl_model.joblib`. The Streamlit app trains its own in-memory model from the CSV, so this file is optional.

## How it works
- Reads historical match data from the CSV.
- Removes records with missing required values and non-decisive/no-result rows.
- Turns each match into two team-perspective examples (team vs opponent).
- One-hot encodes categorical values and trains a Random Forest binary classifier.
- Compares the two teams' scores and displays normalized estimates.

## Important limitations
The supplied data contains only 74 records from one IPL season, including one no-result match. That is far too little data for a reliable real-world prediction system. Test accuracy can change with the split and must not be presented as proof of production quality. The displayed percentages are relative model scores, not calibrated probabilities or betting advice.
