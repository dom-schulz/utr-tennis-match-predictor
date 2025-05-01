import streamlit as st
import pandas as pd
from predict_utils import *
import matplotlib.pyplot as plt
from google.cloud import storage
from google.oauth2 import service_account
import torch
import numpy as np

st.title("UTR Match Predictor 🎾")

with st.sidebar:
    st.header("🔧 Tools & Insights")
    st.markdown("🚧 Tournament Tracker *(coming soon)*")
    st.markdown("🚧 Surface Win Rates *(coming soon)*")

# st.button("Create Custom Player Profile (Coming Soon)", disabled=True)

tabs = st.tabs(["🔮 Predictions", "📅 Upcoming Matches", "📈 Large UTR Moves", "🎾 Player Metrics", "ℹ️ About"])

# ────────────────────────────────────────────────────────────────────────────────
# Load Model & Data
# ────────────────────────────────────────────────────────────────────────────────

credentials_dict = {
        "type": st.secrets["connections_gcs_type"],
        "project_id": st.secrets["connections_gcs_project_id"],
        "private_key_id": st.secrets["connections_gcs_private_key_id"],
        "private_key": st.secrets["connections_gcs_private_key"],
        "client_email": st.secrets["connections_gcs_client_email"],
        "client_id": st.secrets["connections_gcs_client_id"],
        "auth_uri": st.secrets["connections_gcs_auth_uri"],
        "token_uri": st.secrets["connections_gcs_token_uri"],
        "auth_provider_x509_cert_url": st.secrets["connections_gcs_auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["connections_gcs_client_x509_cert_url"],
        "universe_domain": st.secrets["connections_gcs_universe_domain"]
}

@st.cache_resource(show_spinner="🔄  Loading Data & Model from the Cloud...")
def load_everything(credentials_dict):

    # Initialize client (credentials are picked up from st.secrets)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    
    # Initialize the GCS client with credentials and project
    client = storage.Client(credentials=credentials, project=credentials_dict["project_id"])

    # Download model from GCS
    model_bucket = client.bucket(MODEL_BUCKET)
    model_blob = model_bucket.blob(MODEL_BLOB)
    model_bytes = model_blob.download_as_bytes()

    # Load model from bytes
    model = joblib.load(io.BytesIO(model_bytes))
    model.eval()
    
    # Get buckets 
    utr_bucket = client.bucket(UTR_BUCKET)
    matches_bucket = client.bucket(MATCHES_BUCKET)

    # Download data from GCS and return dataframes
    utr_df     = download_csv_from_gcs(credentials_dict, utr_bucket, UTR_FILE)
    matches_df = download_csv_from_gcs(credentials_dict, matches_bucket, MATCHES_FILE)
    
    # Get player history and profiles
    history    = get_player_history(utr_df)
    profiles   = get_player_profiles(matches_df, history)
    
    return model, utr_df, history, profiles


model, utr_df, history, profiles = load_everything(credentials_dict)
player_names = sorted(set(profiles.keys()) & set(history.keys()))


# ────────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ────────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Pick two players")

    # Initialize session state for player selections if not already done
    if 'p1_selection' not in st.session_state:
        st.session_state.p1_selection = None
    if 'p2_selection' not in st.session_state:
        st.session_state.p2_selection = None

    # Define callback functions to update session state
    def update_p1():
        st.session_state.p1_selection = st.session_state.p1_widget
        
    def update_p2():
        st.session_state.p2_selection = st.session_state.p2_widget

    col1, col2 = st.columns(2)
    
    with col1:
        p1 = st.selectbox(
            "Player 1", 
            [""] + player_names,  # Add empty option as first choice
            index=0 if st.session_state.p1_selection is None else 
                 ([""] + player_names).index(st.session_state.p1_selection),
            key="p1_widget",
            on_change=update_p1
        )
    
    with col2:
        # Create full list of options first
        full_p2_options = [""] + player_names
        
        # Only filter out Player 1 if Player 2 isn't already selected
        # or if Player 2 is the same as the new Player 1 selection
        if st.session_state.p2_selection is None or st.session_state.p2_selection == st.session_state.p1_selection:
            p2_options = [""] + [n for n in player_names if n != st.session_state.p1_selection]
            index_p2 = 0
        else:
            # Keep all options and just select the current p2
            p2_options = full_p2_options
            index_p2 = full_p2_options.index(st.session_state.p2_selection)
        
        p2 = st.selectbox(
            "Player 2", 
            p2_options,
            index=index_p2,
            key="p2_widget",
            on_change=update_p2
        )

    # Only proceed with prediction if both players are selected
    if st.session_state.p1_selection and st.session_state.p2_selection:
        # If Player 1 and Player 2 are the same, reset Player 2
        if st.session_state.p1_selection == st.session_state.p2_selection:
            st.session_state.p2_selection = None
            st.write("Please select a different player for Player 2.")
        else:
            p1 = st.session_state.p1_selection
            p2 = st.session_state.p2_selection
            
            # pull latest UTRs
            p1_utr, p2_utr = history[p1], history[p2]

            st.write(f"Current UTRs – **{p1}: {p1_utr:.2f}**, **{p2}: {p2_utr:.2f}**")

            if st.button("Predict"):
                match_stub = {  # minimal dict for preprocess()
                    "p1": p1, "p2": p2, "p1_utr": p1_utr, "p2_utr": p2_utr
                }
                vec = np.array(preprocess_match_data(match_stub, profiles)).reshape(1, -1)
                
                with torch.no_grad():
                    prob = 1 - float(model(torch.tensor(vec, dtype=torch.float32))[0])
                st.metric(label="Probability Player 1 Wins", value=f"{prob*100:0.1f}%")
    else:
        st.write("Please select both players to view UTRs and make a prediction.")

# === Tab: Upcoming Matches ===
with tabs[1]:
    st.header("📅 Upcoming Matches")
    st.subheader("Stay Ahead of the Game")
    st.caption("See what's next on the pro circuit, and who's most likely to rise.")

    st.write("Here you can display upcoming tennis matches (e.g., from a dataset or API).")

    st.markdown("*** Coming Soon ***")

# === Tab: Large UTR Moves ===
with tabs[2]:
    st.header("📈 Large UTR Moves")
    st.subheader("Biggest Shifts in Player Ratings")
    st.caption("Our algorithm tracks the highest-impact UTR swings — who's peaking, who's slipping.")

    st.write("This tab will highlight matches where players gained or lost a large amount of UTR since the previous week.")

    # Load the CSV from your bucket
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    client = storage.Client(credentials=credentials, project=credentials_dict["project_id"])
    utr_bucket = client.bucket(UTR_BUCKET)
    df = download_csv_from_gcs(credentials_dict, utr_bucket, UTR_FILE)

    content = []
    prev_name = ''
    for i in range(len(df)):
        if df['utr'][i] > 13:
            curr_name = df['first_name'][i]+' '+df['last_name'][i]
            if curr_name != prev_name:
                curr_name = df['first_name'][i]+' '+df['last_name'][i]
                content.append([df['first_name'][i]+' '+df['last_name'][i], df['utr'][i+1], df['utr'][i], 
                                df['utr'][i]-df['utr'][i+1], 100*((df['utr'][i]/df['utr'][i+1])-1)])
            prev_name = curr_name
    df = pd.DataFrame(content, columns=["Name", "Previous UTR", "Current UTR", "UTR Change", "UTR % Change"])
    df = df.sort_values(by="UTR % Change", ascending=False)
    st.dataframe(df.head(10))

    df = df.sort_values(by="UTR % Change", ascending=True)
    st.dataframe(df.head(10))

with tabs[3]:
    st.header("🎾 Player Metrics")

    # Load data from GCS
    utr_bucket = client.bucket(UTR_BUCKET)
    df1 = download_csv_from_gcs(credentials_dict, utr_bucket, UTR_FILE)
    matches_bucket = client.bucket(MATCHES_BUCKET)
    df2 = download_csv_from_gcs(credentials_dict, matches_bucket, MATCHES_FILE)

    history = get_player_history_general(df1)
    player_df = get_player_profiles_general(df2, history)

    # Player selection
    st.subheader("Compare Two Players")
    player_names = [""] + sorted(player_df.keys())
    col1, col2 = st.columns(2)

    with col1:
        player1 = st.selectbox("Player 1", player_names, key="player1")

    with col2:
        player2 = st.selectbox("Player 2", player_names, key="player2")

    def display_player_metrics(player1, player2, history):
        if player1 != "" and player2 != "":
            profile = player_df[player1]

            # Assuming you want to take the average of the list if it's a list
            utr_value = profile.get("utr", 0)

            # Check if utr_value is a list and calculate the average if it is
            if isinstance(utr_value, list):
                utr_value = sum(utr_value) / len(utr_value) if utr_value else 0  # Avoid division by zero

            st.markdown(f"### {player1}")
            
            # Limit utr value to 2 decimal places
            utr_value = round(utr_value, 2)
            
            st.metric("Current UTR", utr_value)
            st.metric("Win Rate Vs. Lower UTRs", f"{round(profile.get('win_vs_lower', 0) * 100, 2)}%")
            st.metric("Win Rate Vs. Higher UTRs", f"{round(profile.get('win_vs_higher', 0) * 100, 2)}%")
            st.metric("Win Rate Last 10 Matches", f"{round(profile.get('recent10', 0) * 100, 2)}%")
            try:
                st.metric("Head-To-Head (W-L)", f"{profile.get('h2h')[player2][0]} - {profile.get('h2h')[player2][1]-profile.get('h2h')[player2][0]}")
            except:
                st.metric("Head-To-Head (W-L)", "0 - 0")

    def display_graph(player1, player2, history):
        # Plot both UTR histories
        try:
            utrs1 = history[player1].get("utr", [])
            dates1 = history[player1].get("date", [])
            utrs2 = history[player2].get("utr", [])
            dates2 = history[player2].get("date", [])

            if utrs1 and dates1 and utrs2 and dates2:
                df1 = pd.DataFrame({"Date": pd.to_datetime(dates1), "UTR": utrs1, "Player": player1})
                df2 = pd.DataFrame({"Date": pd.to_datetime(dates2), "UTR": utrs2, "Player": player2})
                df_plot = pd.concat([df1, df2]).sort_values("Date")

                fig, ax = plt.subplots()
                for name, group in df_plot.groupby("Player"):
                    ax.plot(group["Date"], group["UTR"], label=name)  # No marker

                ax.set_title("UTR Over Time")
                ax.set_xlabel("Date")
                ax.set_ylabel("UTR")
                ax.legend()
                ax.grid(True)
                fig.autofmt_xdate()

                st.pyplot(fig)
        except:
            pass

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        display_player_metrics(player1, player2, history)
    with col2:
        display_player_metrics(player2, player1, history)

    st.divider()

    display_graph(player1, player2, history)

with tabs[4]:
    st.markdown("""
    ### 📖 About This Project

    Welcome to the **UTR Tennis Match Predictor** — your go-to tool for analyzing and forecasting professional tennis matches using real data.

    #### 🧠 What We Built  
    Our platform combines historical match outcomes and player UTR (Universal Tennis Rating) data to predict the likelihood of one player winning against another. Under the hood, we use a machine learning model trained on past ATP-level matches, factoring in performance trends, UTR history, and game win ratios.

    #### 🔬 How It Works  
    - We collect and update data from UTR and match databases using web scraping tools.  
    - The predictor uses features like average opponent UTR, win percentages, and recent form.  
    - Users can input two players and instantly receive a match prediction based on model inference.

    #### 📊 Bonus Tools  
    Check out the **Player Metrics** tab to explore individual performance history:
    - UTR progression over time  
    - Win/loss breakdown  
    - Game win percentages  
    - Custom visualizations

    #### 👨‍💻 About the Developers  
    We're a team of student developers and tennis enthusiasts combining our passions for sports analytics, data science, and clean UI design. This is an ongoing project — we're constantly improving predictions, cleaning data, and adding new insights.

    If you have feedback, want to contribute, or just love tennis tech, reach out!
    """)

    st.markdown("💬 We Value Your Feedback!")
    ### Feedback Function ###
    def collect_feedback():
        pass
    # # Create a form to collect feedback
    # with st.form(key="feedback_form"):
    #     # Collect feedback from users
    #     rating = st.slider("How would you rate your experience?", min_value=1, max_value=10)
    #     comments = st.text_area("Any comments or suggestions?", height=150)
        
    #     # Submit button
    #     submit_button = st.form_submit_button(label="Submit Feedback")
        
    #     if submit_button:
    #         # Store the feedback (could also save to a file, database, etc.)
    #         feedback_data = {
    #             "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    #             "rating": rating,
    #             "comments": comments
    #         }
            
    #         # Optionally, save the feedback to a CSV or database
    #         feedback_df = pd.DataFrame([feedback_data])
    #         feedback_df.to_csv("feedback.csv", mode="a", header=False, index=False)
            
    #         # Display thank you message
    #         st.success("Thank you for your feedback!")
    #         st.write("We'll review your comments to improve our platform.")

    collect_feedback()