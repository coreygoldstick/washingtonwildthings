import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime

st.set_page_config(page_title="Umpire Dashboard", layout="wide")

# Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@400;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Roboto Condensed', sans-serif !important;}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Zilla Slab', serif !important;
    font-weight: 700;
    color: var(--text-color);}
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #9B1B30, #1A1A1A);}
section[data-testid="stSidebar"] *:not(.stSelectbox *):not(.stSelectbox label) {
    color: white !important;}
.stSelectbox div[data-baseweb="select"],
.stSelectbox input,
.stSelectbox label {
    color: var(--text-color) !important;
    background-color: var(--background-color) !important;}
div[data-baseweb="radio"] > div, .stSelectbox, .stButton {
    background-color: rgba(255,255,255,0.05);
    padding:6px;
    border-radius:10px;}
div[data-baseweb="radio"] label[data-selected="true"] {
    background-color: white !important;
    color:#9B1B30 !important;
    border-radius:6px;
    padding:4px 8px;
    font-weight:bold;}
.stDataFrame thead tr th {
    background-color:#9B1B30;
    color:white;}
section[data-testid="stSidebar"] label {
    font-weight:bold;
    font-size:1.1em;}
</style>
""", unsafe_allow_html=True)

DATA = "data"
if not os.path.exists(DATA):
    os.makedirs(DATA)

# Team name lookup
team_name_lookup = {
    'WAS_WIL3': 'Washington',
    'EVA_OTT': 'Evansville',
    'JOL_SLA': 'Joliet',
    'LAK_ERI24': 'Lake Erie',
    'SCH_BOO': 'Schaumburg',
    'WIN_CIT29': 'Windy City',
    "FLO_Y'A": 'Florence',
    'OTT_TIT': 'Ottawa',
    'MIS_MUD': 'Mississippi',
    'GAT_GRI': 'Gateway'
}

# Load game files
csv_files = [f for f in os.listdir(DATA) if f.endswith(".csv")]
game_entries = []
for file in csv_files:
    try:
        df = pd.read_csv(os.path.join(DATA, file))
        df.columns = df.columns.str.strip()
        date = pd.to_datetime(
            df.get('Date', pd.Series([file.split('.csv')[0]])).iloc[0], errors='coerce')
        date_str = date.strftime('%B %d, %Y') if not pd.isnull(
            date) else 'Unknown Date'
        teams = df[['PitcherTeam', 'BatterTeam']].iloc[0]
        home = team_name_lookup.get(teams['PitcherTeam'], teams['PitcherTeam'])
        away = team_name_lookup.get(teams['BatterTeam'], teams['BatterTeam'])
        label = f"{date_str} – {home} vs {away}"
        game_entries.append((date, label, file))
    except:
        continue

game_entries = sorted(
    game_entries, key=lambda x: x[0] or pd.Timestamp.max, reverse=True)
game_labels = [entry[1] for entry in game_entries]
file_map = {entry[1]: entry[2] for entry in game_entries}

# Sidebar: Game selector
st.sidebar.title("Game Selection")
selected_game = st.sidebar.selectbox("Select Game", game_labels)

# Load selected game
df = pd.read_csv(os.path.join(DATA, file_map[selected_game]))
df.columns = df.columns.str.strip()
df = df.dropna(subset=['PlateLocSide', 'PlateLocHeight', 'PitchCall'])

# Strike zone dimensions
strike_zone = {
    'left': -0.83,
    'right': 0.83,
    'bottom': 1.5,
    'top': 3.5
}

# Zones
buffer = 0.2

df["InZoneStrict"] = (
    df["PlateLocSide"].between(strike_zone["left"], strike_zone["right"]) &
    df["PlateLocHeight"].between(strike_zone["bottom"], strike_zone["top"])
)

df["InZoneBuffer"] = (
    df["PlateLocSide"].between(strike_zone["left"] - buffer, strike_zone["right"] + buffer) &
    df["PlateLocHeight"].between(
        strike_zone["bottom"] - buffer, strike_zone["top"] + buffer)
)

df["InBufferZone"] = df["InZoneBuffer"] & (~df["InZoneStrict"])

df["CorrectCall"] = (
    ((df["PitchCall"] == "BallCalled") & (~df["InZoneStrict"])) |
    ((df["PitchCall"] == "StrikeCalled") & df["InZoneStrict"])
)

df["MissedCall"] = (
    ((df["PitchCall"] == "BallCalled") & df["InZoneStrict"]) |
    ((df["PitchCall"] == "StrikeCalled") & (~df["InZoneStrict"]))
)

# Counts
total_called_pitches = len(
    df[df["PitchCall"].isin(["BallCalled", "StrikeCalled"])])
total_balls = (df["PitchCall"] == "BallCalled").sum()
total_strikes = (df["PitchCall"] == "StrikeCalled").sum()
correct_calls = df["CorrectCall"].sum()
correct_balls = len(
    df[(df["PitchCall"] == "BallCalled") & (~df["InZoneStrict"])])
correct_strikes = len(
    df[(df["PitchCall"] == "StrikeCalled") & (df["InZoneStrict"])])
buffer_zone_strikes = len(
    df[(df["PitchCall"] == "StrikeCalled") & (df["InBufferZone"])])

# Accuracy
overall_accuracy = correct_calls / \
    total_called_pitches if total_called_pitches > 0 else 0
called_ball_accuracy = correct_balls / total_balls if total_balls > 0 else 0
called_strike_accuracy = correct_strikes / \
    total_strikes if total_strikes > 0 else 0
buffer_zone_accuracy = buffer_zone_strikes / \
    total_strikes if total_strikes > 0 else 0

# Summary Table
summary_df = pd.DataFrame([{
    "Overall Accuracy": f"{overall_accuracy*100:.1f}%",
    "Called Ball Accuracy": f"{called_ball_accuracy*100:.1f}%",
    "Called Strike Accuracy": f"{called_strike_accuracy*100:.1f}%",
    "Strikes in Buffer Zone": f"{buffer_zone_accuracy*100:.1f}%"
}])

st.markdown("### Strike Zone Accuracy")
st.dataframe(summary_df)

# Helper function for name formatting


def format_name(name_str):
    if pd.isna(name_str) or name_str.strip() == "":
        return "Unknown"
    parts = [p.strip() for p in str(name_str).split(",")]
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    else:
        return name_str.strip().title()


# Missed calls DataFrame (reset index)
missed_calls_df = df[df["MissedCall"]].reset_index(drop=True)

# Missed calls dropdown
missed_options = [("All Missed Calls", None)]
for i, (_, row) in enumerate(missed_calls_df.iterrows()):
    pitcher = format_name(row.get("Pitcher", ""))
    batter = format_name(row.get("Batter", ""))

    if row["PitchCall"] == "StrikeCalled":
        desc = "ball called a strike"
    elif row["PitchCall"] == "BallCalled":
        desc = "strike called a ball"
    else:
        desc = "other"

    label = f"{pitcher} to {batter} – {desc}"
    missed_options.append((label, i))

labels = [opt[0] for opt in missed_options]
selected_label = st.selectbox("Review Missed Calls", labels)

# Determine data to plot
selected_idx = next(i for l, i in missed_options if l == selected_label)
if selected_idx is None:
    data_to_plot = missed_calls_df
else:
    data_to_plot = missed_calls_df.iloc[[selected_idx]]

# Plot
st.markdown("### Missed Calls Plot")
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

ax.scatter(
    data_to_plot['PlateLocSide'],
    data_to_plot['PlateLocHeight'],
    color='red',
    edgecolors='black',
    s=150,
    label='Missed Call'
)

rect = Rectangle(
    (strike_zone['left'], strike_zone['bottom']),
    strike_zone['right'] - strike_zone['left'],
    strike_zone['top'] - strike_zone['bottom'],
    linewidth=2, edgecolor='black', facecolor='none'
)
ax.add_patch(rect)

ax.set_xlabel("Horizontal Location (ft)")
ax.set_ylabel("Vertical Location (ft)")
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(1.0, 4.0)
ax.set_aspect('equal')
ax.set_title("Selected Missed Call(s)")
ax.legend()

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.pyplot(fig, use_container_width=True)
