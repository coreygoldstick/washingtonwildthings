import streamlit as st
import pandas as pd
import os
from datetime import datetime
import sys

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
            Use this dashboard to analyze player performance over time. 
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

    st.sidebar.header("Upload New Game CSV Files")
    uploaded_files = st.sidebar.file_uploader(
        "Choose one or more CSV files", type="csv", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA, f"{uploaded_file.name}")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"Saved: {uploaded_file.name}")

    st.sidebar.header("Saved CSV Files")
    saved_files = sorted([f for f in os.listdir(DATA) if f.endswith(".csv")])
    selected_file = None
    if saved_files:
        selected_file = st.sidebar.selectbox(
            "Click a file to preview", saved_files)
    else:
        st.sidebar.write("No files saved yet.")

    if selected_file is not None and os.path.exists(os.path.join(DATA, selected_file)):
        if st.checkbox(f"Show preview of: {selected_file}", value=False):
            preview_df = pd.read_csv(os.path.join(DATA, selected_file))
            st.dataframe(preview_df.head(), use_container_width=True)

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

            hitters = df['Batter'].dropna().unique()
            selected_hitter = st.selectbox("Select a hitter", sorted(hitters))

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
