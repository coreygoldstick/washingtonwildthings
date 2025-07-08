import streamlit as st
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from datetime import date
import networkx as nx

st.set_page_config(
    page_title="Washington Wild Things Performance Dashboard", layout="wide")


def draw_baseball_field(ax):
    theta = np.radians(np.linspace(-45, 45, 100))
    ax.plot([0, -500*np.sin(np.radians(45))],
            [0, 500*np.cos(np.radians(45))], color='gray', linewidth=1)
    ax.plot([0, 500*np.sin(np.radians(45))],
            [0, 500*np.cos(np.radians(45))], color='gray', linewidth=1)
    x = 405 * np.sin(theta)
    y = 405 * np.cos(theta)
    ax.plot(x, y, color='black')
    x_infield = 156 * np.sin(theta)
    y_infield = 156 * np.cos(theta)
    ax.plot(x_infield, y_infield, color='gray', linestyle='--', linewidth=1.5)
    ax.add_patch(plt.Circle((0, 60.5), 3, color='brown'))
    ax.add_patch(plt.Circle((0, 60.5), 13, fill=False,
                 edgecolor='gray', linewidth=2, zorder=6))
    ax.set_xlim(-300, 300)
    ax.set_ylim(0, 500)
    ax.set_aspect('equal')
    ax.axis('off')


def render_left_aligned_table(df, title=None):
    html = ""
    if title:
        html += f"<h4 style='text-align:left; margin-bottom: 0.5em; color: var(--text-color);'>{title}</h4>"
    html += "<table style='width:100%; border-collapse: collapse;'>"
    html += "<thead><tr>"
    for col in df.columns:
        html += f"<th style='text-align:left; padding: 6px; border-bottom: 1px solid #ddd; color: var(--text-color);'>{col}</th>"
    html += "</tr></thead><tbody>"

    for i, (_, row) in enumerate(df.iterrows()):
        is_total_row = str(row.iloc[0]).lower().startswith("<b>total</b>")
        html += "<tr>"
        for val in row:
            val_str = f"<b>{val}</b>" if is_total_row else val
            html += f"<td style='text-align:left; padding: 6px; color: var(--text-color);'>{val_str}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto Condensed', sans-serif !important;}

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Zilla Slab', serif !important;
        font-weight: 700;
        color: var(--text-color);}

    /* (The rest of your styles below remain unchanged) */

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
        background-color: rgba(255, 255, 255, 0.05);
        padding: 6px;
        border-radius: 10px;}

    div[data-baseweb="radio"] label[data-selected="true"] {
        background-color: white !important;
        color: #9B1B30 !important;
        border-radius: 6px;
        padding: 4px 8px;
        font-weight: bold;}

    .stDataFrame thead tr th {
        background-color: #9B1B30;
        color: white;}

    section[data-testid="stSidebar"] label {
        font-weight: bold;
        font-size: 1.1em;}
    </style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'home'

if st.session_state.page == 'home':
    st.image("download.jpg", width=120)
    st.markdown("""
        <h1 style='font-size: 3em; font-weight: bold; margin-top: 0; margin-bottom: 0; color: var(--text-color);'>
            Washington Wild Things Performance Dashboard
        </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p style="font-size: 1.2em; margin-bottom: 1.5em; color: var(--text-color);">
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

stadium_lookup = {
    'WAS_WIL3': 'Wild Things Park',
    'EVA_OTT': 'Bosse Field',
    'JOL_SLA': 'Duly Health and Care Field',
    'LAK_ERI24': 'Crushers Stadium',
    'SCH_BOO': 'Wintrust Field',
    'WIN_CIT29': 'Ozinga Field',
    "FLO_Y'A": 'Thomas More Stadium',
    'OTT_TIT': 'Ottawa Stadium',
    'MIS_MUD': 'Trustmark Park',
    'GAT_GRI': 'Grizzlies Ballpark'}

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
    'GAT_GRI': 'Gateway'}

pitch_type_map = {
    "FourSeamFastball": "Fastball",
    "Four-Seam": "Fastball",
    "Four Seam": "Fastball",
    "Fastball": "Fastball",
    "FourSeamFastBall": "Fastball",
    "2-Seam Fastball": "Sinker",
    "TwoSeamFastball": "Sinker",
    "TwoSeamFastBall": "Sinker",
    "Two-Seam": "Sinker",
    "Sinker": "Sinker",
    "Slider": "Slider",
    "Cutter": "Cutter",
    "Curveball": "Curveball",
    "ChangeUp": "Changeup",
    "Splitter": "Splitter"}

pitch_color_map = {
    'Fastball': '#1f77b4',
    'Slider': '#ff7f0e',
    'Curveball': '#2ca02c',
    'Changeup': '#e377c2',
    'Sinker': '#9467bd',
    'Splitter': '#8c564b',
    'Cutter': '#d62728', }

csv_files = [f for f in os.listdir(DATA) if f.endswith(".csv")]
dataframes = [pd.read_csv(os.path.join(DATA, f)) for f in csv_files]
df_all = pd.concat(dataframes, ignore_index=True)
df_all.columns = df_all.columns.str.strip()
df_all["Batter"] = (df_all["Batter"].astype(str).str.strip().str.replace(r"\s*,\s*", ", ", regex=True).str.title())

wild_things_batter_names = (df_all[df_all['BatterTeam'] == 'WAS_WIL3']['Batter'].dropna().drop_duplicates().sort_values())

batter_display = [f"{name.split(',')[1].strip()} {name.split(',')[0].strip()}" if ',' in name else name
    for name in wild_things_batter_names]

batter_map = {display: original for display, original in zip(batter_display, wild_things_batter_names)}

wild_things_pitcher_names = df_all[df_all['PitcherTeam']== 'WAS_WIL3']['Pitcher'].dropna().unique()
wild_things_pitcher_names = sorted(set(wild_things_pitcher_names))
pitcher_display = [
    f"{name.split(',')[1].strip()} {name.split(',')[0].strip()}" if ',' in name else name for name in wild_things_pitcher_names]
pitcher_map = {display: original for display, original in zip(
    pitcher_display, wild_things_pitcher_names)}

wins, losses = 0, 0
for file in csv_files:
    try:
        game_df = pd.read_csv(os.path.join(DATA, file))
        game_df.columns = game_df.columns.str.strip()

        if {'PitcherTeam', 'BatterTeam', 'RunsScored'}.issubset(game_df.columns):
            teams = game_df[['PitcherTeam', 'BatterTeam']].iloc[0]
            team_raw = teams['PitcherTeam'] if teams['PitcherTeam'].startswith(
                "WAS") else teams['BatterTeam']
            opponent_raw = teams['BatterTeam'] if team_raw == teams['PitcherTeam'] else teams['PitcherTeam']

            team_score = game_df[game_df['PitcherTeam']
                                 == opponent_raw]['RunsScored'].sum()
            opponent_score = game_df[game_df['PitcherTeam']
                                     == team_raw]['RunsScored'].sum()

            if team_score > opponent_score:
                wins += 1
            elif team_score < opponent_score:
                losses += 1
    except:
        continue

adjusted_wins = wins + 3
adjusted_losses = losses + 1
team_record = f"{adjusted_wins}-{adjusted_losses}"

preview_entries = []
for file in csv_files:
    try:
        game_df = pd.read_csv(os.path.join(DATA, file))
        game_df.columns = game_df.columns.str.strip()
        required_cols = {'PitcherTeam', 'BatterTeam', 'Time'}
        if not required_cols.issubset(set(game_df.columns)):
            continue

        first_valid = game_df.dropna(
            subset=['PitcherTeam', 'BatterTeam', 'Time'])
        if first_valid.empty:
            continue

        row = first_valid.iloc[0]
        team1 = team_name_lookup.get(row['PitcherTeam'], row['PitcherTeam'])
        team2 = team_name_lookup.get(row['BatterTeam'], row['BatterTeam'])

        game_date = pd.to_datetime(
            row.get('Date', file.split(".csv")[0]), errors='coerce')
        date_str = game_date.strftime('%B %d, %Y') if not pd.isnull(
            game_date) else "Unknown Date"

        try:
            game_time = pd.to_datetime(str(row['Time'])).strftime('%I:%M %p')
        except:
            game_time = "Unknown Time"

        stadium = stadium_lookup.get(row['PitcherTeam'], "Unknown Ballpark")
        label = f"{date_str} – {team1} vs {team2} @ {stadium} at {game_time}"
        preview_entries.append((game_date, label, file))
    except:
        continue

preview_entries = sorted(preview_entries, key=lambda x: x[0] or pd.Timestamp.max, reverse=True)
preview_labels = [entry[1] for entry in preview_entries]
file_map = {entry[1]: entry[2] for entry in preview_entries}

st.sidebar.title("Player Selection")
position = st.sidebar.radio("Select Position", ["Hitter", "Pitcher"], horizontal=True, key="position_radio", index=None)

selected_player = ""
display_name = ""

if position == "Hitter":
    display_options = [""] + batter_display
    display_name = st.sidebar.selectbox("Select a Hitter", display_options, key="hitter_select")
    selected_player = batter_map.get(display_name, "")
elif position == "Pitcher":
    display_options = [""] + pitcher_display
    display_name = st.sidebar.selectbox("Select a Pitcher", display_options, key="pitcher_select")
    selected_player = pitcher_map.get(display_name, "")
else:
    selected_player = None

if position is None or selected_player == "":
    st.markdown("# Select a position and player from the sidebar to view data.")
    st.stop()

if selected_player != "":
    selected_game = st.sidebar.selectbox(
        "Filter by Game", ["All Games"] + preview_labels)
    today = datetime.today().strftime("%B %d, %Y")

    st.sidebar.markdown(
        f"""
        <hr style='margin-top: 20px; margin-bottom: 10px; border: 1px solid #ffffff22;'>

        <div style='text-align: center; color: #dddddd; font-size: 0.9em;'>
            <b>Wild Things record:</b> {team_record}<br>
            <b>Date:</b> {today}
        </div>
        """,
        unsafe_allow_html=True)

    if selected_game != "All Games":
        df = pd.read_csv(os.path.join(DATA, file_map[selected_game]))
    else:
        df = df_all.copy()
    df["Batter"] = (df["Batter"].astype(str).str.strip().str.replace(r"\s*,\s*", ", ", regex=True).str.title())

else:
    df = df_all.copy()

if 'TaggedPitchType' in df.columns:
    df['PitchTypeUsed'] = df['TaggedPitchType'].where(
        df['TaggedPitchType'].str.lower() != 'undefined', df['AutoPitchType'])
else:
    df['PitchTypeUsed'] = df['AutoPitchType']

df['PitchTypeUsed'] = df['PitchTypeUsed'].replace(pitch_type_map)

if position == "Hitter":
    hitter_df = df[df['Batter'] == selected_player]

    if hitter_df.empty:
        st.markdown(
            f"<h2>Hitter Summary: {display_name}</h2>", unsafe_allow_html=True)
        st.warning("No data available for this hitter in the selected game.")
        st.stop()

    side_series = hitter_df["BatterSide"].dropna()
    if not side_series.empty:
        side = side_series.iloc[0]
        bats_label = "RHB" if side == "Right" else "LHB"
    else:
        bats_label = ""

    st.markdown(
        f"<h2>Hitter Summary: {display_name} <span style='color:gray;'>{bats_label}</span></h2>", unsafe_allow_html=True)

    if 'PitcherThrows' in df.columns:
        handedness = st.radio("Filter by Pitcher Handedness", [
                              'All', 'Right', 'Left'])
        if handedness != 'All':
            hitter_df = hitter_df[hitter_df['PitcherThrows'] == handedness]

    pitches_seen = hitter_df.groupby(
        'PitchTypeUsed').size().reset_index(name='Pitches Seen')

    batted = hitter_df[hitter_df['PlayResult'].isin(
        ['Single', 'Double', 'Triple', 'HomeRun', 'Out'])]
    if batted.empty:
        st.warning("No batted ball events available for this hitter.")
        st.stop()

    ab_df = hitter_df[
        (hitter_df['TaggedHitType'].str.lower() != 'undefined') |
        (hitter_df['KorBB'] == 'Strikeout')
    ]
    total_abs = len(ab_df)
    total_hits = ab_df['PlayResult'].isin(
        ['Single', 'Double', 'Triple', 'HomeRun']).sum()
    regular_avg = round(total_hits / total_abs, 3) if total_abs > 0 else 0

    pitch_stats = batted.groupby('PitchTypeUsed').agg(
        BIP=('PlayResult', lambda x: x.isin(
            ['Single', 'Double', 'Triple', 'HomeRun', 'Out']).sum()),
        Hits=('PlayResult', lambda x: x.isin(
            ['Single', 'Double', 'Triple', 'HomeRun']).sum()),
        AvgEV=('ExitSpeed', 'mean'),
        AvgLA=('Angle', 'mean')
    ).reset_index()

    swings_df = hitter_df[hitter_df["PitchCall"].isin(
        ["StrikeSwinging", "InPlay", "FoulBall", "FoulBallFieldable", "FoulBallNotFieldable"])]
    whiffs_df = hitter_df[hitter_df["PitchCall"] == "StrikeSwinging"]

    swings_count = swings_df.groupby("PitchTypeUsed").size().rename("Swings")
    whiffs_count = whiffs_df.groupby("PitchTypeUsed").size().rename("Whiffs")

    pitch_stats = pitch_stats.merge(
        pitches_seen, on='PitchTypeUsed', how='outer')
    pitch_stats = pitch_stats.merge(
        swings_count, on='PitchTypeUsed', how='left')
    pitch_stats = pitch_stats.merge(
        whiffs_count, on='PitchTypeUsed', how='left')

    pitch_stats.fillna({"Hits": 0, "BIP": 0, "Pitches Seen": 0,
                       "Swings": 0, "Whiffs": 0}, inplace=True)

    pitch_stats['Whiff %'] = round(
        pitch_stats['Whiffs'] / pitch_stats['Swings'].replace(0, np.nan), 3).fillna(0)
    pitch_stats['BABIP'] = round(
        pitch_stats['Hits'] / pitch_stats['BIP'].replace(0, np.nan), 3).fillna(0)
    pitch_stats['Batting Average'] = round(
        pitch_stats['Hits'] / pitch_stats['Pitches Seen'].replace(0, np.nan), 3).fillna(0)

    display_df = pitch_stats.rename(columns={
        "PitchTypeUsed": "Pitch Type",
        "Pitches Seen": "Pitches Seen",
        "BIP": "Balls Hit Into Play",
        "Hits": "Hits",
        "AvgEV": "Avg Exit Velo (mph)",
        "AvgLA": "Avg Launch Angle (°)",
        "Whiff %": "Whiff %",
        "BABIP": "BABIP",
        "Batting Average": "Batting Average"
    })

    display_df["Avg Exit Velo (mph)"] = display_df["Avg Exit Velo (mph)"].round(
        2)
    display_df["Avg Launch Angle (°)"] = display_df["Avg Launch Angle (°)"].round(
        2)
    display_df["Whiff %"] = display_df["Whiff %"].apply(
        lambda x: f"{x * 100:.1f}%")
    display_df["Balls Hit Into Play"] = display_df["Balls Hit Into Play"].astype(
        int)
    display_df["Hits"] = display_df["Hits"].astype(int)

    display_df = display_df[[
        "Pitch Type",
        "Pitches Seen",
        "Balls Hit Into Play",
        "Hits",
        "BABIP",
        "Batting Average",
        "Avg Exit Velo (mph)",
        "Avg Launch Angle (°)",
        "Whiff %",
    ]]

    totals = {
        "Pitch Type": "<b>Total</b>",
        "Pitches Seen": display_df["Pitches Seen"].sum(),
        "Balls Hit Into Play": display_df["Balls Hit Into Play"].sum(),
        "Hits": display_df["Hits"].sum(),
        "Avg Exit Velo (mph)": display_df["Avg Exit Velo (mph)"].mean().round(2),
        "Avg Launch Angle (°)": display_df["Avg Launch Angle (°)"].mean().round(2),
        "BABIP": round(display_df["Hits"].sum() / display_df["Balls Hit Into Play"].sum(), 3) if display_df["Balls Hit Into Play"].sum() else 0,
        "Whiff %": f"{round(display_df['Whiff %'].str.rstrip('%').astype(float).mean(), 1)}%",
        "Batting Average": f"{regular_avg:.3f}"
    }

    display_df = pd.concat(
        [display_df, pd.DataFrame([totals])], ignore_index=True)
    render_left_aligned_table(display_df, title="Pitch Type Breakdown")

    avg_by_outs = (ab_df.groupby("Outs")
                   .apply(lambda x: x["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum() / len(x))
                   .round(3)
                   .reset_index(name="AVG")
                   )

    avg_by_outs["Outs"] = avg_by_outs["Outs"].astype(int).astype(str)

    st.markdown("### Batting Average by Outs")
    render_left_aligned_table(avg_by_outs)

    hits = hitter_df[hitter_df['PlayResult'].isin(
        ['Single', 'Double', 'Triple', 'HomeRun'])]
    hits = hits.dropna(subset=['Distance', 'Bearing', 'PitchTypeUsed'])
    hits['x'] = hits['Distance'] * np.sin(np.radians(hits['Bearing']))
    hits['y'] = hits['Distance'] * np.cos(np.radians(hits['Bearing']))

    pitch_types = hits['PitchTypeUsed'].unique()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Hits Spray Chart (Pitch Type)")
        fig, ax = plt.subplots(figsize=(6, 6))

        for r in [100, 200, 300, 400]:
            circle = plt.Circle(
                (0, 0), r, color='orange', linestyle='-', linewidth=1.2, fill=False, alpha=0.3)
            ax.add_patch(circle)
            ax.text(0, r + 5, str(r), ha='center',
                    va='bottom', fontsize=8, color='orange')

        draw_baseball_field(ax)

        for ptype in pitch_types:
            subset = hits[hits['PitchTypeUsed'] == ptype]
            color = pitch_color_map.get(ptype, 'gray')
            ax.scatter(subset['x'], subset['y'], label=ptype,
                       color=color, s=40, edgecolors='black', zorder=5)

        ax.legend(title='Pitch Type', loc='upper left',
                  fontsize='small', title_fontsize='medium')
        st.pyplot(fig)

    with col2:
        st.markdown("### Hits Spray Chart (Hit Type)")
        fig2, ax2 = plt.subplots(figsize=(6, 6))

        for r in [100, 200, 300, 400]:
            ring = plt.Circle((0, 0), r, color='orange',
                              linestyle='-', linewidth=1.2, fill=False, alpha=0.3)
            ax2.add_patch(ring)
            ax2.text(0, r + 5, str(r), ha='center',
                     va='bottom', fontsize=8, color='orange')

        draw_baseball_field(ax2)

        hit_color_map = {
            'Single': '#17becf',
            'Double': '#bcbd22',
            'Triple': '#ff9896',
            'HomeRun': '#d62728'}

        for result in ['Single', 'Double', 'Triple', 'HomeRun']:
            subset = hits[hits['PlayResult'] == result]
            ax2.scatter(subset['x'], subset['y'], label=result,
                        color=hit_color_map[result], s=40, edgecolors='black', zorder=5)

        ax2.legend(title='Hit Result', loc='upper left',
                   fontsize='small', title_fontsize='medium')
        st.pyplot(fig2)

        strike_zone = {
            'left': -0.85,
            'right': 0.85,
            'bottom': 1.5,
            'top': 3.5
        }

        outside_zone = hitter_df[
            (hitter_df['PlateLocSide'] < strike_zone['left']) |
            (hitter_df['PlateLocSide'] > strike_zone['right']) |
            (hitter_df['PlateLocHeight'] < strike_zone['bottom']) |
            (hitter_df['PlateLocHeight'] > strike_zone['top'])
        ]

        swings = ["StrikeSwinging", "FoulBall", "InPlay",
                  "FoulBallFieldable", "FoulBallNotFieldable"]
        chases = outside_zone[outside_zone["PitchCall"].isin(swings)]

        chase_pct = len(chases) / \
            len(outside_zone) if len(outside_zone) > 0 else 0

        st.markdown("### Chases by Pitch Type")
        st.markdown(
            f"<b>Chase Percentage:</b> {chase_pct:.1%}", unsafe_allow_html=True)
        chase_df = hitter_df[hitter_df['PitchCall'].isin(["StrikeSwinging", "InPlay", "FoulBall", "FoulBallFieldable", "FoulBallNotFieldable"])].dropna(
            subset=["PlateLocSide", "PlateLocHeight", "PitchTypeUsed"])

        outside_zone = chase_df[
            (chase_df["PlateLocSide"].abs() > 0.85) |
            (chase_df["PlateLocHeight"] < 1.5) |
            (chase_df["PlateLocHeight"] > 3.5)]

        fig_chase, ax_chase = plt.subplots(figsize=(6, 6))
        unique_ptypes = sorted(outside_zone["PitchTypeUsed"].dropna().unique())
        colors = plt.cm.get_cmap('tab10', len(unique_ptypes))
        ptype_color_map = {ptype: colors(i)
                           for i, ptype in enumerate(unique_ptypes)}

        for ptype in unique_ptypes:
            subset = outside_zone[outside_zone["PitchTypeUsed"] == ptype]
            ax_chase.scatter(
                subset["PlateLocSide"], subset["PlateLocHeight"],
                label=ptype,
                color=ptype_color_map[ptype],
                edgecolors='black', s=60)

        strike_zone = Rectangle(
            (-0.85, 1.5), 1.7, 2.0, linewidth=2, edgecolor='black', facecolor='none')
        ax_chase.add_patch(strike_zone)

        ax_chase.set_title(
            "Swings Outside the Strike Zone by Pitch Type (Hitter's view)")
        ax_chase.set_xlabel("Horizontal Location (ft)")
        ax_chase.set_ylabel("Vertical Location (ft)")
        ax_chase.set_xlim(2, -2)
        ax_chase.set_ylim(0, 5)
        ax_chase.legend(title="Pitch Type", fontsize='small',
                        title_fontsize='medium')
        st.pyplot(fig_chase)

if position == "Pitcher":
    selected_pitcher = selected_player

    if selected_pitcher:
        pidf_all = df[df["Pitcher"] == selected_pitcher].copy()

        pitcher_rows = df[df["Pitcher"] == selected_player]
        if not pitcher_rows.empty and "PitcherThrows" in pitcher_rows.columns:
            hand_series = pitcher_rows["PitcherThrows"].dropna()
            if not hand_series.empty:
                hand = hand_series.iloc[0]
                throws_label = "RHP" if hand == "Right" else "LHP"
            else:
                throws_label = ""
        else:
            throws_label = ""

        batter_handedness = st.session_state.get(
            "pitcher_batter_handedness", "All")

        batter_filter_label = ""
        st.markdown(
            f"<h2>Pitcher Summary: {display_name} <span style='color:gray;'>{throws_label}{batter_filter_label}</span></h2>", unsafe_allow_html=True)

        pidf = pidf_all.copy()
        df["PitchofPA"] = pd.to_numeric(df["PitchofPA"], errors="coerce")
        hits = pidf["PlayResult"].isin(
            ["Single", "Double", "Triple", "HomeRun"]).sum()
        runs = pidf["RunsScored"].sum()
        walks = pidf["KorBB"].eq("Walk").sum()
        strikeouts = pidf["KorBB"].eq("Strikeout").sum()

        pidf["OutsOnPlay"] = pd.to_numeric(
            pidf["OutsOnPlay"], errors="coerce").fillna(0)

        outs_on_play = pidf["OutsOnPlay"].sum()

        strikeout_outs = pidf[(pidf["OutsOnPlay"] == 0) & (
            pidf["KorBB"] == "Strikeout")].shape[0]

        total_outs = outs_on_play + strikeout_outs

        innings_pitched = total_outs / 3

        results_df = pd.DataFrame({
            "Innings Pitched": [f"{innings_pitched:.1f}"],
            "Hits": [hits],
            "Runs": [runs],
            "Walks": [walks],
            "Strikeouts": [strikeouts]})

        render_left_aligned_table(results_df)

        hand_filter_placeholder = st.empty()
        batter_handedness = hand_filter_placeholder.radio("Filter by Batter Handedness", [
                                                          'All', 'Right', 'Left'], horizontal=True, key="pitcher_batter_handedness")

        if batter_handedness != 'All':
            pidf = pidf_all[pidf_all["BatterSide"] == batter_handedness].copy()
        else:
            pidf = pidf_all.copy()

        df["PitchofPA"] = pd.to_numeric(df["PitchofPA"], errors="coerce")
        hits = pidf["PlayResult"].isin(
            ["Single", "Double", "Triple", "HomeRun"]).sum()
        runs = pidf["RunsScored"].sum()
        walks = pidf["KorBB"].eq("Walk").sum()
        strikeouts = pidf["KorBB"].eq("Strikeout").sum()

        pidf["OutsOnPlay"] = pd.to_numeric(
            pidf["OutsOnPlay"], errors="coerce").fillna(0)

        outs_on_play = pidf["OutsOnPlay"].sum()

        strikeout_outs = pidf[(pidf["OutsOnPlay"] == 0) & (
            pidf["KorBB"] == "Strikeout")].shape[0]

        total_outs = outs_on_play + strikeout_outs

        innings_pitched = total_outs / 3

        results_df = pd.DataFrame({
            "Innings Pitched": [f"{innings_pitched:.1f}"],
            "Hits": [hits],
            "Runs": [runs],
            "Walks": [walks],
            "Strikeouts": [strikeouts]})

        if pidf.empty:
            st.warning(
                "No data available for this pitcher in the selected game.")
            st.stop()

        metrics_df = pidf.dropna(
            subset=["PitchTypeUsed", "RelSpeed", "SpinRate", "InducedVertBreak", "HorzBreak"])

        avg_metrics = metrics_df.groupby("PitchTypeUsed").agg(
            Count=("PitchTypeUsed", "count"),
            Velocity=("RelSpeed", lambda x: f"{x.mean():.2f} mph"),
            SpinRate=("SpinRate", lambda x: f"{x.mean():.0f} rpm"),
            VerticalBreak=("InducedVertBreak",
                           lambda x: f"{x.mean():.2f} in."),
            HorizontalBreak=("HorzBreak", lambda x: f"{x.mean():.2f} in.")
        ).reset_index().rename(columns={
            "PitchTypeUsed": "Pitch Type",
            "Count": "Number of Pitches",
            "VerticalBreak": "Vertical Break",
            "HorizontalBreak": "Horizontal Break"})

        if batter_handedness != 'All':
            pidf = pidf[pidf["BatterSide"] == batter_handedness]

        batters_faced = pidf[pidf["PitchofPA"] == 1].shape[0]
        total_pitches = len(pidf)

        first_pitch_calls = pidf[pidf["PitchofPA"] == 1]["PitchCall"]
        total_pas = len(first_pitch_calls)
        first_pitch_strikes = first_pitch_calls.isin(
            ["StrikeCalled", "StrikeSwinging", "FoulBall", "InPlay", "FoulBallFieldable", "FoulBallNotFieldable"]).sum()
        first_pitch_strike_pct = first_pitch_strikes / total_pas if total_pas else 0

        strike_count = pidf[pidf["PitchCall"].isin(
            ["StrikeCalled", "StrikeSwinging", "FoulBall", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"])]
        strike_pct = len(strike_count) / total_pitches if total_pitches else 0

        swings = pidf[pidf["PitchCall"].isin(
            ["StrikeSwinging", "InPlay", "FoulBall"])]
        whiffs = pidf[pidf["PitchCall"] == "StrikeSwinging"]
        whiff_pct = len(whiffs) / len(swings) if len(swings) else 0

        in_play = pidf[pidf["PitchCall"] == "InPlay"]
        groundballs = in_play[in_play["TaggedHitType"].str.contains(
            "Ground", case=False, na=False)]
        groundball_pct = len(groundballs) / len(in_play) if len(in_play) else 0

        summary_df = pd.DataFrame({
            "Batters Faced": [batters_faced],
            "Total Pitches": [total_pitches],
            "Strike %": [f"{strike_pct:.1%}"],
            "1st Pitch Strike %": [f"{first_pitch_strike_pct:.1%}"],
            "Whiff %": [f"{whiff_pct:.1%}"],
            "Ground Ball %": [f"{groundball_pct:.1%}"]})

        render_left_aligned_table(summary_df)

        st.markdown("### Average Pitch Metrics by Pitch Type")
        st.markdown(
            "<div style='font-size: 0.85em; color: gray; margin-bottom: 0.5em;'>"
            "<b>Note:<b> Pitch types with very few occurrences may be misclassified (e.g., a sinker labeled as a fastball)."
            "</div>", unsafe_allow_html=True)
        render_left_aligned_table(avg_metrics)

        strike_zone_df = pidf.dropna(
            subset=["PlateLocSide", "PlateLocHeight", "PitchCall"])

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("### Pitch Locations Heatmap")
            reds = plt.cm.get_cmap('Reds')
            darker_reds = LinearSegmentedColormap.from_list(
                "darker_reds", reds(np.linspace(0.3, 1.0, 256)))
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            hb = ax1.hexbin(
                strike_zone_df["PlateLocSide"], strike_zone_df["PlateLocHeight"], gridsize=30, cmap=darker_reds, bins='log')
            ax1.set_title("All Pitch Locations (Pitcher's View)")
            ax1.set_xlabel("Horizontal Location (ft)")
            ax1.set_ylabel("Vertical Location (ft)")
            ax1.set_xlim(-2, 2)
            ax1.set_ylim(0, 5)
            strike_zone = Rectangle(
                (-0.85, 1.5), 1.7, 2.0, linewidth=2, edgecolor='black', facecolor='none')
            ax1.add_patch(strike_zone)
            st.pyplot(fig1)

            st.markdown("### Hits Allowed")
            fig_hits_pitch, ax_hits_pitch = plt.subplots(figsize=(6, 6))

            for r in [100, 200, 300, 400]:
                circle = plt.Circle(
                    (0, 0), r, color='orange', linestyle='-', linewidth=1.2, fill=False, alpha=0.3)
                ax_hits_pitch.add_patch(circle)
                ax_hits_pitch.text(0, r + 5, str(r), ha='center',
                                   va='bottom', fontsize=8, color='orange')

            draw_baseball_field(ax_hits_pitch)

            hits = pidf[pidf['PlayResult'].isin(
                ['Single', 'Double', 'Triple', 'HomeRun'])].copy()
            hits = hits.dropna(subset=['Distance', 'Bearing', 'PitchTypeUsed'])
            hits['x'] = hits['Distance'] * np.sin(np.radians(hits['Bearing']))
            hits['y'] = hits['Distance'] * np.cos(np.radians(hits['Bearing']))

            hit_color_map = {
                'Single': '#17becf',
                'Double': '#bcbd22',
                'Triple': '#ff9896',
                'HomeRun': '#d62728'}

            for result in ['Single', 'Double', 'Triple', 'HomeRun']:
                subset = hits[hits['PlayResult'] == result]
                ax_hits_pitch.scatter(subset['x'], subset['y'], label=result,
                                      color=hit_color_map[result], s=40, edgecolors='black', zorder=5)

            ax_hits_pitch.set_xlim(-300, 300)
            ax_hits_pitch.set_ylim(0, 500)
            ax_hits_pitch.set_aspect('equal')
            ax_hits_pitch.axis('off')
            ax_hits_pitch.legend(
                title='Hit Type', loc='upper left', fontsize='small', title_fontsize='medium')
            st.pyplot(fig_hits_pitch)

            st.markdown("### Pitch Mix by Count")
            count_df = pidf.dropna(
                subset=["Balls", "Strikes", "PitchTypeUsed"]).copy()
            count_df["Count"] = count_df["Balls"].astype(int).astype(
                str) + "-" + count_df["Strikes"].astype(int).astype(str)

            count_order = ["0-0", "1-0", "0-1", "2-0", "1-1",
                           "0-2", "3-0", "2-1", "1-2", "3-1", "2-2", "3-2"]
            count_df = count_df[count_df["Count"].isin(count_order)]

            count_pie_data = count_df.groupby(
                ["Count", "PitchTypeUsed"]).size().unstack(fill_value=0)

            count_pie_props = count_pie_data.div(
                count_pie_data.sum(axis=1), axis=0)

            positions = {
                "0-0": (3, 6),
                "1-0": (4, 5),
                "0-1": (2, 5),
                "2-0": (5, 4),
                "1-1": (3, 4),
                "0-2": (1, 4),
                "3-0": (6, 3),
                "2-1": (4, 3),
                "1-2": (2, 3),
                "3-1": (5, 2),
                "2-2": (3, 2),
                "3-2": (4, 1),
            }

            fig, ax = plt.subplots(figsize=(6, 6))
            G = nx.DiGraph()

            edges = [
                ("0-0", "1-0"), ("0-0", "0-1"),
                ("1-0", "2-0"), ("1-0", "1-1"),
                ("0-1", "1-1"), ("0-1", "0-2"),
                ("2-0", "3-0"), ("2-0", "2-1"),
                ("1-1", "2-1"), ("1-1", "1-2"),
                ("0-2", "1-2"),
                ("2-1", "3-1"), ("2-1", "2-2"),
                ("1-2", "2-2"),
                ("2-2", "3-2"),
            ]

            G.add_edges_from(edges)

            for edge in G.edges:
                start, end = edge
                x_vals = [positions[start][0], positions[end][0]]
                y_vals = [positions[start][1], positions[end][1]]
                ax.plot(x_vals, y_vals, color='gray', linewidth=1.5, zorder=1)

            for count, (x, y) in positions.items():
                if count in count_pie_props.index:
                    sizes = count_pie_props.loc[count].values
                    labels = count_pie_props.columns.tolist()
                    total = sizes.sum()
                    if total == 0:
                        continue
                    start = 0
                    for i, size in enumerate(sizes):
                        if size == 0:
                            continue
                        wedge = plt.matplotlib.patches.Wedge(
                            (x, y), 0.4, start * 360, (start + size) * 360, facecolor=pitch_color_map.get(labels[i], "gray"), edgecolor="black")
                        ax.add_patch(wedge)
                        start += size

                    ax.text(x, y + 0.55, count, ha='center',
                            va='bottom', fontsize=8, color='gray')

            ax.set_xlim(0, 7)
            ax.set_ylim(0, 7)
            ax.axis("off")

            for i, ptype in enumerate(count_pie_props.columns):
                ax.scatter([], [], color=pitch_color_map.get(
                    ptype, "gray"), label=ptype)
            ax.legend(title="Pitch Type", loc="lower left",
                      fontsize=8, title_fontsize=9)

            st.pyplot(fig)

        with right_col:
            st.markdown("### Pitch Locations by Pitch Type")
            fig_pitch_types, ax_pitch_types = plt.subplots(figsize=(6, 6))

            unique_pitch_types = pidf['PitchTypeUsed'].dropna().unique()

            colors = plt.cm.get_cmap('tab10', len(unique_pitch_types))
            pitch_color_map = {ptype: colors(
                i) for i, ptype in enumerate(unique_pitch_types)}

            for ptype in unique_pitch_types:
                subset = pidf[pidf['PitchTypeUsed'] == ptype]
                ax_pitch_types.scatter(
                    subset['PlateLocSide'], subset['PlateLocHeight'],
                    label=ptype, color=pitch_color_map[ptype],
                    edgecolors='black', s=60
                )

            strike_zone = Rectangle(
                (-0.85, 1.5), 1.7, 2.0, linewidth=2, edgecolor='black', facecolor='none')
            ax_pitch_types.add_patch(strike_zone)

            ax_pitch_types.set_title(
                "All Pitch Locations by Pitch Type (Pitcher's View)")
            ax_pitch_types.set_xlabel("Horizontal Location (ft)")
            ax_pitch_types.set_ylabel("Vertical Location (ft)")
            ax_pitch_types.set_xlim(-2, 2)
            ax_pitch_types.set_ylim(0, 5)
            ax_pitch_types.legend(title="Pitch Type",
                                  fontsize='small', title_fontsize='medium')
            st.pyplot(fig_pitch_types)

            st.markdown("### Hits Allowed by Pitch Type")
            fig_hits_pitch, ax_hits_pitch = plt.subplots(figsize=(6, 6))
            for r in [100, 200, 300, 400]:
                circle = plt.Circle(
                    (0, 0), r, color='orange', linestyle='-', linewidth=1.2, fill=False, alpha=0.3)
                ax_hits_pitch.add_patch(circle)
                ax_hits_pitch.text(0, r + 5, str(r), ha='center',
                                   va='bottom', fontsize=8, color='orange')

            draw_baseball_field(ax_hits_pitch)

            for ptype in hits['PitchTypeUsed'].dropna().unique():
                subset = hits[hits['PitchTypeUsed'] == ptype]
                color = pitch_color_map.get(ptype, 'gray')
                ax_hits_pitch.scatter(
                    subset['x'], subset['y'], label=ptype, color=color, s=40, edgecolors='black', zorder=5)

            ax_hits_pitch.set_xlim(-300, 300)
            ax_hits_pitch.set_ylim(0, 500)
            ax_hits_pitch.set_aspect('equal')
            ax_hits_pitch.axis('off')
            ax_hits_pitch.legend(
                title='Pitch Type', loc='upper left', fontsize='small', title_fontsize='medium')
            st.pyplot(fig_hits_pitch)

            st.markdown("### Whiffs by Pitch Type")
            whiffs = strike_zone_df[strike_zone_df["PitchCall"]
                                    == "StrikeSwinging"]

            unique_pitch_types = sorted(
                whiffs["PitchTypeUsed"].dropna().unique())
            selected_types = unique_pitch_types

            colors = plt.cm.get_cmap('tab10', len(unique_pitch_types))
            pitch_color_map = {ptype: colors(
                i) for i, ptype in enumerate(unique_pitch_types)}

            fig2, ax2 = plt.subplots(figsize=(6, 6))
            for ptype in selected_types:
                subset = whiffs[whiffs["PitchTypeUsed"] == ptype]
                ax2.scatter(subset["PlateLocSide"], subset["PlateLocHeight"],
                            label=ptype, color=pitch_color_map[ptype], edgecolors='black', s=60)

            strike_zone = Rectangle(
                (-0.85, 1.5), 1.7, 2.0, linewidth=2, edgecolor='black', facecolor='none')
            ax2.add_patch(strike_zone)

            ax2.set_title("Swings and Misses by Pitch Type (Pitcher's view)")
            ax2.set_xlabel("Horizontal Location (ft)")
            ax2.set_ylabel("Vertical Location (ft)")
            ax2.set_xlim(-2, 2)
            ax2.set_ylim(0, 5)
            ax2.legend(title="Pitch Type", fontsize='small',
                       title_fontsize='medium')
            st.pyplot(fig2)

    else:
        st.info("Please select a pitcher to view stats.")

st.markdown(
    """
    <div style='font-size: 0.8em; color: gray;'>
    <b>Disclaimer:</b> Stats shown are based on TrackMan data from most, but not all, Wild Things games in 2025 and may not exactly match official team or league records. 
    </div>
    """, unsafe_allow_html=True)
