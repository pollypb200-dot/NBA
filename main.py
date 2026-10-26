import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

# --- Global Variables ---
model = None
le = None
team_abbr_to_id = {}


def train_model():
    global model, le, team_abbr_to_id

    print("1. Fetching NBA teams...")
    nba_teams = teams.get_teams()
    team_abbr_to_id = {team['abbreviation']: team['id'] for team in nba_teams}

    print("2. Downloading historical game data (Please wait)...")
    all_games = pd.DataFrame()
    for team in nba_teams:
        team_id = team['id']
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
        games = gamefinder.get_data_frames()[0]
        all_games = pd.concat([all_games, games], ignore_index=True)
        time.sleep(0.5)

    print("3. Feature Engineering: Adding Rest Days & Head-to-Head...")
    all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
    all_games = all_games.sort_values(by=['TEAM_ID', 'GAME_DATE'])

    # Base Stats
    all_games['WIN'] = all_games['WL'].apply(lambda x: 1 if x == 'W' else 0)
    all_games['PTS'] = all_games['PTS'].astype(float)
    all_games['PPG'] = all_games.groupby('TEAM_ID')['PTS'].transform('mean')
    all_games['LAST_GAME'] = all_games.groupby('TEAM_ID')['WIN'].shift(1).fillna(0)

    # Calculate Rest Days (Days between games minus 1. Back-to-Back = 0 rest days)
    # We cap it at 7 days so off-season gaps don't skew the data
    all_games['REST_DAYS'] = (all_games.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days - 1)
    all_games['REST_DAYS'] = all_games['REST_DAYS'].fillna(3).clip(lower=0, upper=7)

    home_games = all_games[all_games['MATCHUP'].str.contains('vs.')].copy()
    away_games = all_games[all_games['MATCHUP'].str.contains('@')].copy()

    home_games = home_games[['GAME_ID', 'TEAM_ID', 'PPG', 'LAST_GAME', 'REST_DAYS', 'WIN']]
    home_games.columns = ['GAME_ID', 'HOME_TEAM_ID', 'HOME_PPG', 'HOME_LAST_GAME', 'HOME_REST_DAYS', 'HOME_WIN']

    away_games = away_games[['GAME_ID', 'TEAM_ID', 'PPG', 'LAST_GAME', 'REST_DAYS']]
    away_games.columns = ['GAME_ID', 'AWAY_TEAM_ID', 'AWAY_PPG', 'AWAY_LAST_GAME', 'AWAY_REST_DAYS']

    matchups = pd.merge(home_games, away_games, on='GAME_ID').dropna()

    print("4. Encoding and Training XGBoost Model...")
    le = LabelEncoder()
    all_ids = pd.concat([matchups['HOME_TEAM_ID'], matchups['AWAY_TEAM_ID']]).unique()
    le.fit(all_ids)

    matchups['HOME_TEAM_ID_ENC'] = le.transform(matchups['HOME_TEAM_ID'])
    matchups['AWAY_TEAM_ID_ENC'] = le.transform(matchups['AWAY_TEAM_ID'])

    # New Features Array including Rest Days
    X = matchups[['HOME_TEAM_ID_ENC', 'AWAY_TEAM_ID_ENC', 'HOME_PPG', 'AWAY_PPG',
                  'HOME_LAST_GAME', 'AWAY_LAST_GAME', 'HOME_REST_DAYS', 'AWAY_REST_DAYS']]
    y = matchups['HOME_WIN']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Using XGBoost instead of Random Forest
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model Training Complete! XGBoost Accuracy: {accuracy * 100:.2f}%. Launching interface...")


# --- GUI Application Logic ---
def parse_rest_days(value):
    """Parse a rest-days selection into a number.

    Accepts plain numbers ("0".."7") and legacy "4+" style values.
    Training caps rest days at 7, so clamp to the same 0-7 range.
    """
    cleaned = str(value).strip().rstrip("+")
    return min(max(float(cleaned), 0), 7)


def make_prediction():
    if model is None:
        messagebox.showerror("Error", "Model is still training.")
        return

    try:
        home_abbr = home_team_var.get()
        away_abbr = away_team_var.get()

        if home_abbr == away_abbr:
            messagebox.showwarning("Matchup Error", "A team cannot play itself.")
            return

        home_ppg = float(home_pts_var.get())
        away_ppg = float(away_pts_var.get())
        home_last = 1 if home_form_var.get() == "Win" else 0
        away_last = 1 if away_form_var.get() == "Win" else 0

        home_rest = parse_rest_days(home_rest_var.get())
        away_rest = parse_rest_days(away_rest_var.get())

        home_id_enc = le.transform([team_abbr_to_id[home_abbr]])[0]
        away_id_enc = le.transform([team_abbr_to_id[away_abbr]])[0]

        new_data = pd.DataFrame({
            'HOME_TEAM_ID_ENC': [home_id_enc],
            'AWAY_TEAM_ID_ENC': [away_id_enc],
            'HOME_PPG': [home_ppg],
            'AWAY_PPG': [away_ppg],
            'HOME_LAST_GAME': [home_last],
            'AWAY_LAST_GAME': [away_last],
            'HOME_REST_DAYS': [home_rest],
            'AWAY_REST_DAYS': [away_rest]
        })

        predictions = model.predict(new_data)
        probabilities = model.predict_proba(new_data)

        home_win_prob = probabilities[0][1] * 100
        away_win_prob = probabilities[0][0] * 100

        winner = home_abbr if predictions[0] == 1 else away_abbr
        result_text = f"Predicted Winner: {winner}\n\n"
        result_text += f"{home_abbr} (Home) Win Probability: {home_win_prob:.2f}%\n"
        result_text += f"{away_abbr} (Away) Win Probability: {away_win_prob:.2f}%"

        result_label.config(text=result_text, foreground="green")

    except ValueError:
        messagebox.showerror("Input Error", "Please ensure PPG and Rest Days are valid numbers.")


# 1. Initialize & Train
train_model()

# 2. Setup Tkinter Window
root = tk.Tk()
root.title("Advanced XGBoost NBA Predictor")
root.geometry("500x520")
root.configure(padx=20, pady=20)

team_list = sorted(list(team_abbr_to_id.keys()))

# --- HOME TEAM UI ---
tk.Label(root, text="HOME TEAM", font=("Arial", 10, "bold")).grid(row=0, column=0, pady=(0, 10), sticky="w")

home_team_var = tk.StringVar()
ttk.Combobox(root, textvariable=home_team_var, values=team_list, state="readonly", width=12).grid(row=1, column=0,
                                                                                                  sticky="w")

tk.Label(root, text="Home PPG:").grid(row=2, column=0, sticky="w")
home_pts_var = tk.StringVar(value="115.0")
ttk.Entry(root, textvariable=home_pts_var, width=15).grid(row=3, column=0, sticky="w", pady=(0, 5))

tk.Label(root, text="Home Last Game:").grid(row=4, column=0, sticky="w")
home_form_var = tk.StringVar(value="Win")
ttk.Combobox(root, textvariable=home_form_var, values=["Win", "Loss"], state="readonly", width=12).grid(row=5, column=0,
                                                                                                        sticky="w",
                                                                                                        pady=(0, 5))

tk.Label(root, text="Home Rest Days (0 for B2B):").grid(row=6, column=0, sticky="w")
home_rest_var = tk.StringVar(value="1")
ttk.Combobox(root, textvariable=home_rest_var, values=["0", "1", "2", "3", "4", "5", "6", "7"], state="readonly", width=12).grid(
    row=7, column=0, sticky="w")

# --- AWAY TEAM UI ---
tk.Label(root, text="AWAY TEAM", font=("Arial", 10, "bold")).grid(row=0, column=1, pady=(0, 10), sticky="w")

away_team_var = tk.StringVar()
ttk.Combobox(root, textvariable=away_team_var, values=team_list, state="readonly", width=12).grid(row=1, column=1,
                                                                                                  sticky="w")

tk.Label(root, text="Away PPG:").grid(row=2, column=1, sticky="w")
away_pts_var = tk.StringVar(value="108.5")
ttk.Entry(root, textvariable=away_pts_var, width=15).grid(row=3, column=1, sticky="w", pady=(0, 5))

tk.Label(root, text="Away Last Game:").grid(row=4, column=1, sticky="w")
away_form_var = tk.StringVar(value="Loss")
ttk.Combobox(root, textvariable=away_form_var, values=["Win", "Loss"], state="readonly", width=12).grid(row=5, column=1,
                                                                                                        sticky="w",
                                                                                                        pady=(0, 5))

tk.Label(root, text="Away Rest Days (0 for B2B):").grid(row=6, column=1, sticky="w")
away_rest_var = tk.StringVar(value="1")
ttk.Combobox(root, textvariable=away_rest_var, values=["0", "1", "2", "3", "4", "5", "6", "7"], state="readonly", width=12).grid(
    row=7, column=1, sticky="w")

# --- PREDICT BUTTON & RESULTS ---
ttk.Button(root, text="Run XGBoost Simulation", command=make_prediction).grid(row=8, column=0, columnspan=2, pady=25,
                                                                              ipadx=10, ipady=5)

result_label = ttk.Label(root, text="Enter stats and simulate.", justify="center", font=("Arial", 12, "bold"))
result_label.grid(row=9, column=0, columnspan=2)

root.mainloop()