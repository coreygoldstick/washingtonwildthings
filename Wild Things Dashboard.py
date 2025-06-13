import streamlit as st
import pandas as pd
import os
from datetime import datetime
import sys
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

stadium_lookup = {
    'WAS_WIL3': 'Wild Things Park',
    'EVA_OTT': 'Bosse Field',
    'x': 'Grizzlies Ballpark',
    'JOL_SLA': 'Duly Health and Care Field',
    'LAK_ERI24': 'Crushers Stadium',
    'SCH_BOO': 'Wintrust Field',
    'WIN_CIT29': 'Ozinga Field',
    "FLO_Y'A": 'Thomas More Stadium',
    'OTT_TIT': 'Ottawa Stadium',
    'x': 'Trustmark Park',
    'x': 'Skylands Stadium',
    'x': 'Hinchliffe Stadium',
    'x': 'Stade CANAC',
    'x': 'Joseph L. Bruno Stadium',
    'x': 'Grainger Stadium'
}

team_name_lookup = {
    'WAS_WIL3': 'Washington',
    'EVA_OTT': 'Evansville',
    'x': 'Gateway',
    'JOL_SLA': 'Joliet',
    'LAK_ERI24': 'Lake Erie',
    'SCH_BOO': 'Schaumburg',
    'WIN_CIT29': 'Windy City',
    "FLO_Y'A": 'Florence',
    'OTT_TIT': 'Ottawa',
    'x': 'Mississippi',
    'x': 'Sussex County',
    'x': 'New Jersey',
    'x': 'Québec',
    'x': 'Tri-City',
    'x': 'Down East'
}

st.set_page_config(
    page_title="Washington Wild Things Performance Dashboard", layout="wide")

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
            Start by clicking the button below.
        </p>
    """, unsafe_allow_html=True)

    if st.button("Enter Dashboard"):
        st.session_state.page = 'dashboard'
        st.rerun()

    st.stop()

if st.session_state.page == 'dashboard':
    DATA = "data"
    if not os.path.exists(DATA):
        os.makedirs(DATA)

    saved_files = sorted([f for f in os.listdir(DATA) if f.endswith(".csv")])
    preview_entries = []

    for file in saved_files:
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

            first_row = first_valid.iloc[0]
            team1_raw = first_row['PitcherTeam']
            team2_raw = first_row['BatterTeam']
            team1 = team_name_lookup.get(team1_raw, team1_raw)
            team2 = team_name_lookup.get(team2_raw, team2_raw)

            game_date = pd.to_datetime(first_row.get(
                'Date', file.split(".csv")[0]), errors='coerce')
            date_str = game_date.strftime('%B %d, %Y') if not pd.isnull(
                game_date) else "Unknown Date"

            raw_time = str(first_row['Time'])
            try:
                game_time = pd.to_datetime(raw_time).strftime('%I:%M %p')
            except:
                game_time = "Unknown Time"

            stadium = stadium_lookup.get(team1_raw, "Unknown Ballpark")

            label = f"{team1} vs {team2} @ {stadium} – {date_str} at {game_time}"
            preview_entries.append((game_date, label, file))
        except Exception:
            continue

    preview_entries = sorted(
        preview_entries, key=lambda x: x[0] or pd.Timestamp.max)
    preview_labels = [entry[1] for entry in preview_entries]
    file_map = {entry[1]: entry[2] for entry in preview_entries}

    if preview_labels:
        selected_label = st.sidebar.selectbox(
            "Click a game to preview", preview_labels)
        selected_file = file_map[selected_label]
        if st.checkbox(f"Show preview of: {selected_label}", value=False):
            preview_df = pd.read_csv(os.path.join(DATA, selected_file))
            preview_df.columns = preview_df.columns.str.strip()

            required_cols = {'Batter', 'Pitcher',
                             'PitcherTeam', 'BatterTeam', 'RunsScored'}
            if not required_cols.issubset(preview_df.columns):
                st.warning("Preview data is missing necessary columns.")
            else:
                teams = preview_df[['PitcherTeam', 'BatterTeam']].iloc[0]
                team1_raw, team2_raw = teams['PitcherTeam'], teams['BatterTeam']
                team1 = team_name_lookup.get(team1_raw, team1_raw)
                team2 = team_name_lookup.get(team2_raw, team2_raw)

                team1_score = preview_df[preview_df['PitcherTeam']
                                         == team2_raw]['RunsScored'].sum()
                team2_score = preview_df[preview_df['PitcherTeam']
                                         == team1_raw]['RunsScored'].sum()

                st.subheader("Final Score")
                st.markdown(
                    f"""
                    <div style="font-size: 1.2em; line-height: 1.6;">
                        <strong>{team_name_lookup.get(team1_raw, team1_raw)}:</strong> {int(team1_score)}<br>
                        <strong>{team_name_lookup.get(team2_raw, team2_raw)}:</strong> {int(team2_score)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                def get_lineup(df, team_raw_name):
                    team_df = df[df['BatterTeam'] == team_raw_name]
                    lineup = team_df.drop_duplicates(subset='Batter')[
                        'Batter'].tolist()[:9]
                    lineup = [
                        f"{i+1}. {b.split(',')[0].strip()}" for i, b in enumerate(lineup)]

                    pitcher_series = df[df['PitcherTeam']
                                        == team_raw_name]['Pitcher'].dropna()
                    starting_pitcher = pitcher_series.iloc[0] if not pitcher_series.empty else "Unknown"
                    sp_last = starting_pitcher.split(",")[0].strip()
                    lineup.append(f"SP: {sp_last}")

                    return pd.DataFrame({team_name_lookup.get(team_raw_name, team_raw_name): lineup})

                st.write("")

                col1, col2 = st.columns(2)
                with col1:
                    lineup1 = get_lineup(preview_df, team1_raw)
                    st.dataframe(lineup1, hide_index=True,
                                 use_container_width=True)

                with col2:
                    lineup2 = get_lineup(preview_df, team2_raw)
                    st.dataframe(lineup2, hide_index=True,
                                 use_container_width=True)

    csv_files = [f for f in os.listdir(DATA) if f.endswith(".csv")]
    dataframes = [pd.read_csv(os.path.join(DATA, f)) for f in csv_files]

    if dataframes:
        df = pd.concat(dataframes, ignore_index=True)

        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
            min_date, max_date = df['game_date'].min(), df['game_date'].max()
            date_range = st.sidebar.date_input(
                "Filter by game date", [min_date, max_date])

            if len(date_range) == 2:
                df = df[(df['game_date'] >= pd.to_datetime(date_range[0])) &
                        (df['game_date'] <= pd.to_datetime(date_range[1]))]

        st.sidebar.header("View Options")
        view = st.sidebar.selectbox(
            "Select view type", ["Hitter Summary", "Pitcher Summary"])

        if view == "Hitter Summary":
            st.subheader("Hitter Summary")

            hitters = [''] + sorted(df['Batter'].dropna().unique())
            selected_hitter = st.selectbox("Select a hitter", hitters)

            if selected_hitter == '':
                st.info("Please select a hitter to view stats.")
            else:
                if isinstance(selected_hitter, str) and "," in selected_hitter:
                    last, first = selected_hitter.split(",", 1)
                    display_name = f"{first.strip()} {last.strip()}"
                else:
                    display_name = selected_hitter

                hitter_df = df[df['Batter'] == selected_hitter]

                if 'PitcherThrows' in df.columns:
                    split = st.radio("Pitcher Handedness", [
                        'All', 'Right', 'Left'], key="handedness_filter")
                if split != 'All':
                    hitter_df = hitter_df[hitter_df['PitcherThrows'] == split]

                if hitter_df.empty:
                    st.warning(
                        f"No batted ball data available for {display_name} against selected handedness.")
                else:
                    required_cols = ['AutoPitchType',
                                     'PlayResult', 'ExitSpeed', 'Angle']
                if all(col in hitter_df.columns for col in required_cols):
                    def calculate_batting_stats(group):
                        in_play = group['PlayResult'].isin(
                            ['Single', 'Double', 'Triple', 'HomeRun', 'Out'])
                        batted = group[in_play]

                        hits = batted['PlayResult'].isin(
                            ['Single', 'Double', 'Triple', 'HomeRun']).sum()
                        total = len(batted)
                        avg = hits / total if total else 0

                        tb_map = {'Single': 1, 'Double': 2,
                                  'Triple': 3, 'HomeRun': 4}
                        total_bases = batted['PlayResult'].map(
                            tb_map).fillna(0).sum()
                        slg = total_bases / total if total else 0

                        return pd.Series({
                            'Number of balls in Play': total,
                            'Number of Hits': hits,
                            'AVG on Balls in Play': round(avg, 3),
                            'SLG on Balls in Play': round(slg, 3),
                            'Avg Exit Velo (mph)': round(batted['ExitSpeed'].mean(), 3),
                            'Avg Launch Angle (°)': round(batted['Angle'].mean(), 3)
                        })

                    pitch_stats = (
                        hitter_df
                        .groupby('AutoPitchType')[required_cols]
                        .apply(calculate_batting_stats)
                        .reset_index()
                    )

                    pitch_stats = pitch_stats.dropna(subset=['AutoPitchType'])

                    totals_row = pd.Series({
                        'AutoPitchType': 'Totals/AVGs',
                        'Number of balls in Play': pitch_stats['Number of balls in Play'].sum(),
                        'Number of Hits': pitch_stats['Number of Hits'].sum(),
                        'AVG on Balls in Play': round((pitch_stats['AVG on Balls in Play'] * pitch_stats['Number of balls in Play']).sum() / pitch_stats['Number of balls in Play'].sum(), 3),
                        'SLG on Balls in Play': round((pitch_stats['SLG on Balls in Play'] * pitch_stats['Number of balls in Play']).sum() / pitch_stats['Number of balls in Play'].sum(), 3),
                        'Avg Exit Velo (mph)': round((pitch_stats['Avg Exit Velo (mph)'] * pitch_stats['Number of balls in Play']).sum() / pitch_stats['Number of balls in Play'].sum(), 1),
                        'Avg Launch Angle (°)': round((pitch_stats['Avg Launch Angle (°)'] * pitch_stats['Number of balls in Play']).sum() / pitch_stats['Number of balls in Play'].sum(), 1)
                    })

                    pitch_stats = pd.concat(
                        [pitch_stats, totals_row.to_frame().T], ignore_index=True)

                    pitch_stats = pitch_stats.rename(
                        columns={"AutoPitchType": "Pitch Type"})
                    st.markdown(
                        f"### Pitch Type Breakdown for Hitter: {display_name}")
                    st.dataframe(
                        pitch_stats, use_container_width=True, hide_index=True)
                else:
                    st.warning(
                        "No batted ball data available for this hitter against selected handedness.")

                hits = hitter_df[hitter_df['PlayResult'].isin(
                    ['Single', 'Double', 'Triple', 'HomeRun'])]
                hits = hits.dropna(
                    subset=['Distance', 'Bearing', 'AutoPitchType'])
                hits['x'] = hits['Distance'] * \
                    np.sin(np.radians(hits['Bearing']))
                hits['y'] = hits['Distance'] * \
                    np.cos(np.radians(hits['Bearing']))

                fig, ax = plt.subplots(figsize=(6, 6))

                for r in [100, 200, 300, 400, 500]:
                    circle = plt.Circle(
                        (0, 0), r, color='orange', linestyle='-', linewidth=1.2, fill=False, alpha=0.3)
                    ax.add_patch(circle)
                    ax.text(0, r + 5, str(r), ha='center',
                            va='bottom', fontsize=8, color='orange')

                ax.set_xlim(-250, 250)
                ax.set_ylim(0, 500)
                ax.set_aspect('equal', adjustable='box')
                ax.axis('off')

                pitch_types = hits['AutoPitchType'].unique()
                colors = plt.cm.get_cmap('tab10', len(pitch_types))
                pitch_color_map = {ptype: colors(
                    i) for i, ptype in enumerate(pitch_types)}

                for ptype in pitch_types:
                    subset = hits[hits['AutoPitchType'] == ptype]
                    ax.scatter(subset['x'], subset['y'], label=ptype,
                               color=pitch_color_map[ptype], s=40, edgecolors='black')

                ax.legend(title='Pitch Type', loc='upper left',
                          fontsize='small', title_fontsize='medium')
                left_col, _ = st.columns([1, 1.5])
                with left_col:
                    st.markdown("### Spray Chart (Pitch Type)")
                    st.pyplot(fig)

        elif view == "Pitcher Summary":
            st.subheader("Pitcher Summary coming soon!")
        else:
            st.info("No data yet. Please upload a CSV to get started.")

    st.markdown(
        "<div style='font-size: 0.8em; color: gray;'>"
        "<b>Disclaimer:</b> The statistics displayed on this dashboard are based solely on TrackMan data collected from most, but not all Wild Things games this season. While the data is accurate, it may not exactly match official team or league stats. All metrics reflect only the games for which TrackMan data was available."
        "</div>",
        unsafe_allow_html=True
    )
