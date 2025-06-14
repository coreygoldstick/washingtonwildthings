import streamlit as st
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Washington Wild Things Performance Dashboard", layout="wide")

st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #9B1B30, #1A1A1A);
        color: white !important;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    div[data-baseweb="radio"] > div, .stSelectbox, .stButton {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 6px;
        border-radius: 10px;
    }

    div[data-baseweb="radio"] label[data-selected="true"] {
        background-color: white !important;
        color: #9B1B30 !important;
        border-radius: 6px;
        padding: 4px 8px;
        font-weight: bold;
    }

    h1, h2, h3 {
        color: #9B1B30;
    }

    .stDataFrame thead tr th {
        background-color: #9B1B30;
        color: white;

    section[data-testid="stSidebar"] label {
        font-weight: bold;
        font-size: 1.1em;
    }
    </style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'home'

if st.session_state.page == 'home':
    st.image("download.jpg", width=120)

    st.markdown("""
        <h1 style='font-size: 3em; font-weight: bold; margin-top: 0; margin-bottom: 0;'>
            Washington Wild Things Performance Dashboard
        </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p style="font-size: 1.2em; margin-bottom: 1.5em;">
            Use this dashboard to analyze player performance.
        </p>
    """, unsafe_allow_html=True)

    if st.button("Enter Dashboard"):
        st.session_state.page = 'dashboard'
        st.rerun()

    st.stop()

DATA = "data"
if not os.path.exists(DATA):
    os.makedirs(DATA)

stadium_lookup = {
    'WAS_WIL3': 'Wild Things Park',
    'EVA_OTT': 'Bosse Field',
    'JOL_SLA': 'Duly Health and Care Field',
    'LAK_ERI24': 'Crushers Stadium',
    'SCH_BOO': 'Wintrust Field',
    'WIN_CIT29': 'Ozinga Field',
    "FLO_Y'A": 'Thomas More Stadium',
    'OTT_TIT': 'Ottawa Stadium'
}

team_name_lookup = {
    'WAS_WIL3': 'Washington',
    'EVA_OTT': 'Evansville',
    'JOL_SLA': 'Joliet',
    'LAK_ERI24': 'Lake Erie',
    'SCH_BOO': 'Schaumburg',
    'WIN_CIT29': 'Windy City',
    "FLO_Y'A": 'Florence',
    'OTT_TIT': 'Ottawa'
}

pitch_type_map = {
    "FourSeamFastball": "Fastball",
    "Four-Seam": "Fastball",
    "Four Seam": "Fastball",
    "Fastball": "Fastball",
    "FourSeamFastBall": "Fastball",
    "2-Seam Fastball": "Sinker",
    "TwoSeamFastball": "Sinker",
    "Two-Seam": "Sinker",
    "Sinker": "Sinker",
    "Slider": "Slider",
    "Cutter": "Cutter",
    "Curveball": "Curveball",
    "ChangeUp": "Changeup",
    "Splitter": "Splitter"
}

csv_files = [f for f in os.listdir(DATA) if f.endswith(".csv")]
dataframes = [pd.read_csv(os.path.join(DATA, f)) for f in csv_files]
df_all = pd.concat(dataframes, ignore_index=True)
df_all.columns = df_all.columns.str.strip()

preview_entries = []
for file in csv_files:
    try:
        game_df = pd.read_csv(os.path.join(DATA, file))
        game_df.columns = game_df.columns.str.strip()
        required_cols = {'PitcherTeam', 'BatterTeam', 'Time'}
        if not required_cols.issubset(set(game_df.columns)):
            continue

        first_valid = game_df.dropna(subset=['PitcherTeam', 'BatterTeam', 'Time'])
        if first_valid.empty:
            continue

        row = first_valid.iloc[0]
        team1 = team_name_lookup.get(row['PitcherTeam'], row['PitcherTeam'])
        team2 = team_name_lookup.get(row['BatterTeam'], row['BatterTeam'])

        game_date = pd.to_datetime(row.get('Date', file.split(".csv")[0]), errors='coerce')
        date_str = game_date.strftime('%B %d, %Y') if not pd.isnull(game_date) else "Unknown Date"

        try:
            game_time = pd.to_datetime(str(row['Time'])).strftime('%I:%M %p')
        except:
            game_time = "Unknown Time"

        stadium = stadium_lookup.get(row['PitcherTeam'], "Unknown Ballpark")
        label = f"{team1} vs {team2} @ {stadium} – {date_str} at {game_time}"
        preview_entries.append((game_date, label, file))
    except:
        continue

preview_entries = sorted(preview_entries, key=lambda x: x[0] or pd.Timestamp.max)
preview_labels = [entry[1] for entry in preview_entries]
file_map = {entry[1]: entry[2] for entry in preview_entries}

st.sidebar.title("Player Selection")
position = st.sidebar.radio("Select Position", ["Hitter", "Pitcher"], horizontal=True, index=None)
selected_player = ""

if position == "Hitter":
    players = [""] + sorted(df_all['Batter'].dropna().unique())
    selected_player = st.sidebar.selectbox("Select a Hitter", players)
elif position == "Pitcher":
    players = [""] + sorted(df_all['Pitcher'].dropna().unique())
    selected_player = st.sidebar.selectbox("Select a Pitcher", players)
else:
    selected_player = None

if position is None or selected_player == "":
    st.markdown("## Please select a position and player from the sidebar to load data.")
    st.stop()

if selected_player != "":
    selected_game = st.sidebar.selectbox("Filter by Game", ["All Games"] + preview_labels)
    if selected_game != "All Games":
        df = pd.read_csv(os.path.join(DATA, file_map[selected_game]))
    else:
        df = df_all.copy()
else:
    df = df_all.copy()

if 'TaggedPitchType' in df.columns:
    df['PitchTypeUsed'] = df['TaggedPitchType'].where(
        df['TaggedPitchType'].str.lower() != 'undefined',
        df['AutoPitchType']  
    )
else:
    df['PitchTypeUsed'] = df['AutoPitchType']

df['PitchTypeUsed'] = df['PitchTypeUsed'].replace(pitch_type_map)

if position == "Hitter":
    st.header(f"Hitter Summary: {selected_player}")
    hitter_df = df[df['Batter'] == selected_player]

    if hitter_df.empty:
        st.warning("No data available for this hitter.")
        st.stop()

    if 'PitcherThrows' in df.columns:
        handedness = st.radio("Filter by Pitcher Handedness", ['All', 'Right', 'Left'])
        if handedness != 'All':
            hitter_df = hitter_df[hitter_df['PitcherThrows'] == handedness]

    batted = hitter_df[hitter_df['PlayResult'].isin(['Single','Double','Triple','HomeRun','Out'])]
    if batted.empty:
        st.warning("No batted ball events available for this hitter.")
        st.stop()

    pitch_stats = batted.groupby('PitchTypeUsed').agg(
        BIP=('PlayResult', 'count'),
        Hits=('PlayResult', lambda x: x.isin(['Single','Double','Triple','HomeRun']).sum()),
        AvgEV=('ExitSpeed', 'mean'),
        AvgLA=('Angle', 'mean')
    ).reset_index()
    
    pitch_stats['AVG'] = round(pitch_stats['Hits'] / pitch_stats['BIP'], 3)

    display_df = pitch_stats.rename(columns={"AVG": "Average", "PitchTypeUsed": "Pitch Type", "BIP": "Balls Hit Into Play", "AvgEV": "Avg Exit Velo (mph)", "AvgLA": "Avg Launch Angle (°)"})

    display_df["Avg Exit Velo (mph)"] = display_df["Avg Exit Velo (mph)"].round(2)
    display_df["Avg Launch Angle (°)"] = display_df["Avg Launch Angle (°)"].round(2)
    
    st.subheader("Batted Ball Performance")
    st.dataframe(display_df, hide_index=True)
    
    def draw_baseball_field(ax):
        theta = np.radians(np.linspace(-45, 45, 100))
        ax.plot([0, -500*np.sin(np.radians(45))], [0, 500*np.cos(np.radians(45))], color='gray', linewidth=1)
        ax.plot([0, 500*np.sin(np.radians(45))], [0, 500*np.cos(np.radians(45))], color='gray', linewidth=1)
        x = 405 * np.sin(theta)
        y = 405 * np.cos(theta)
        ax.plot(x, y, color='black')
        x_infield = 156 * np.sin(theta)
        y_infield = 156 * np.cos(theta)
        ax.plot(x_infield, y_infield, color='gray', linestyle='--', linewidth=1.5)
        ax.add_patch(plt.Circle((0, 60.5), 3, color='brown'))
        ax.set_xlim(-250, 250)
        ax.set_ylim(0, 500)
        ax.set_aspect('equal')
        ax.axis('off')

    hits = hitter_df[hitter_df['PlayResult'].isin(['Single', 'Double', 'Triple', 'HomeRun'])]
    hits = hits.dropna(subset=['Distance', 'Bearing', 'PitchTypeUsed'])
    hits['x'] = hits['Distance'] * np.sin(np.radians(hits['Bearing']))
    hits['y'] = hits['Distance'] * np.cos(np.radians(hits['Bearing']))

    pitch_types = hits['PitchTypeUsed'].unique()
    colors = plt.cm.get_cmap('tab10', len(pitch_types))
    pitch_color_map = {ptype: colors(i) for i, ptype in enumerate(pitch_types)}

    left_col, _ = st.columns([1, 1.5])
    with left_col:
        st.markdown("### Hits Spray Chart (Pitch Type)")
        fig, ax = plt.subplots(figsize=(6, 6))

        for r in [100, 200, 300, 400]:
            circle = plt.Circle((0, 0), r, color='orange', linestyle='-', linewidth=1.2, fill=False, alpha=0.3)
            ax.add_patch(circle)
            ax.text(0, r + 5, str(r), ha='center', va='bottom', fontsize=8, color='orange')

        draw_baseball_field(ax)

        for ptype in pitch_types:
            subset = hits[hits['PitchTypeUsed'] == ptype]
            ax.scatter(subset['x'], subset['y'], label=ptype, color=pitch_color_map[ptype], s=40, edgecolors='black', zorder=5)

        ax.legend(title='Pitch Type', loc='upper left', fontsize='small', title_fontsize='medium')
        st.pyplot(fig)

elif position == "Pitcher":
    st.header(f"Pitcher Summary: {selected_player}")
    pitcher_df = df[df['Pitcher'] == selected_player]
    pitcher_df["PitchofPA"] = pd.to_numeric(pitcher_df["PitchofPA"], errors="coerce")

    total_pitches = len(pitcher_df)
    batters_faced = pitcher_df[pitcher_df["PitchofPA"] == 1].shape[0]
    strikes = pitcher_df[pitcher_df["PitchCall"].isin(["StrikeCalled", "StrikeSwinging", "FoulBall"])]
    whiffs = pitcher_df[pitcher_df["PitchCall"] == "StrikeSwinging"]
    in_play = pitcher_df[pitcher_df["PitchCall"] == "InPlay"]
    grounders = in_play[in_play["TaggedHitType"] == "GroundBall"]

    summary = pd.DataFrame({
        "Batters Faced": [batters_faced],
        "Total Pitches": [total_pitches],
        "Strike %": [f"{len(strikes)/total_pitches:.1%}" if total_pitches else "N/A"],
        "Whiff %": [f"{len(whiffs)/len(pitcher_df[pitcher_df['PitchCall'].isin(['StrikeSwinging','InPlay','FoulBall'])]) if len(pitcher_df) else 0:.1%}"],
        "Ground Ball %": [f"{len(grounders)/len(in_play) if len(in_play) else 0:.1%}"]
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

st.markdown(
    """
    <div style='font-size: 0.8em; color: gray;'>
    <b>Disclaimer:</b> The statistics displayed on this dashboard are based solely on TrackMan data collected from most, but not all Wild Things games during the 2025 season. While the data is accurate, it may not exactly match official team or league stats. 
    <br><b>Note:</b> Any data shown for players not on the Washington Wild Things reflects only their performance in games against the Wild Things.
    </div>
    """,
    unsafe_allow_html=True
)
