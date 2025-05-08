import streamlit as st
import pandas as pd
from predict_utils import *
from google.cloud import storage
from google.oauth2 import service_account
from wordcloud import WordCloud
import torch
import numpy as np
from bs4 import BeautifulSoup
import requests


st.title("Universal Tennis Predictions 🎾")

with st.sidebar:
    st.header("🔧 Tools & Insights")
    st.markdown("🚧 Tournament Tracker *(coming soon)*")
    st.markdown("🚧 Surface Win Rates *(coming soon)*")

# st.button("Create Custom Player Profile (Coming Soon)", disabled=True)

tabs = st.tabs(["🔮 Predictions", "📅 Upcoming Matches", "📈 Large UTR Moves", "🏆 ATP Rankings", "ℹ️ About"])

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
    graph_hist = get_player_history_general(utr_df)
    profiles   = get_set_player_profiles(matches_df, history, st=st)
    
    return model, utr_df, history, profiles, graph_hist

# Define custom color function
def color_func(word, **kwargs):
    return color_map.get(word, "black")


model, utr_df, history, profiles, graph_hist = load_everything(credentials_dict)
player_names = sorted(set(profiles.keys()) & set(history.keys()))


# ────────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ────────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Pick two players")

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
        p1 = st.selectbox("Player 1", [""] + player_names, 
        index=0 if st.session_state.p1_selection is None else
            ([""] + player_names).index(st.session_state.p1_selection),
            key='p1_widget',
            on_change=update_p1)
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
                # match_stub = {  # minimal dict for preprocess()
                #     "p1": p1, "p2": p2, "p1_utr": p1_utr, "p2_utr": p2_utr
                # }
                vec = preprocess_player_data(p1, p2, profiles)
                
                with torch.no_grad():
                    prob = model(torch.tensor(vec, dtype=torch.float32))[0]
                    if prob >= 0.5:
                        winner = p1
                    else:
                        winner = p2
                st.metric(label="Winner", value=winner)
    else:
        st.write("Please select both players to view UTRs and make a prediction.")

    # st.divider()
    
    # ==================== Graph ======================== #

    display_graph(p1, p2, graph_hist) # Graph

    # st.divider()

    # =================== Metrics ======================== #

    col1, col2 = st.columns(2)
    with col1:
        display_player_metrics(p1, p2, history, profiles)
    with col2:
        display_player_metrics(p2, p1, history, profiles)
                    
# === Tab: Upcoming Matches ===
with tabs[1]:
    st.header("📅 Upcoming Matches")
    st.subheader("Stay Ahead of the Game")
    st.caption("See what's next on the pro circuit, and who's most likely to rise.")

    html1 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-08T09:00:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60370623" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60370623" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-08T09:00:00+00:00" data-format="hh:mm">02:00</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-08T09:00:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R128</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60370623/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/matteo-arnaldi-sr-competitor-505550/" title="Matteo Arnaldi" data-id="sr:competitor:505550" data-slug="matteo-arnaldi-sr-competitor-505550" aria-label="Matteo Arnaldi"><div class="tc-player"><small class="tc-player__country">ITA</small> <span class="tc-player__name">M. <span>Arnaldi</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/roberto-bautista-agut-sr-competitor-16720/" title="Roberto Bautista Agut" data-id="sr:competitor:16720" data-slug="roberto-bautista-agut-sr-competitor-16720" aria-label="Roberto Bautista Agut"><div class="tc-player"><small class="tc-player__country">ESP</small> <span class="tc-player__name">R. <span>Bautista Agut</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">M. <strong>Arnaldi</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">73.8%</span></div></div></div></a></div>"""
    html2 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-08T12:10:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60370619" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60370619" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-08T12:10:00+00:00" data-format="hh:mm">05:10</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-08T12:10:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R128</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60370619/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/luca-nardi-sr-competitor-450477/" title="Luca Nardi" data-id="sr:competitor:450477" data-slug="luca-nardi-sr-competitor-450477" aria-label="Luca Nardi"><div class="tc-player"><small class="tc-player__country">ITA</small> <span class="tc-player__name">L. <span>Nardi</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/flavio-cobolli-sr-competitor-399637/" title="Flavio Cobolli" data-id="sr:competitor:399637" data-slug="flavio-cobolli-sr-competitor-399637" aria-label="Flavio Cobolli"><div class="tc-player"><small class="tc-player__country">ITA</small> <span class="tc-player__name">F. <span>Cobolli</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">F. <strong>Cobolli</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">65.2%</span></div></div></div></a></div>"""
    html3 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-08T17:00:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60370635" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60370635" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-08T17:00:00+00:00" data-format="hh:mm">10:00</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-08T17:00:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R128</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60370635/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/jacob-fearnley-sr-competitor-407783/" title="Jacob Fearnley" data-id="sr:competitor:407783" data-slug="jacob-fearnley-sr-competitor-407783" aria-label="Jacob Fearnley"><div class="tc-player"><small class="tc-player__country">GBR</small> <span class="tc-player__name">J. <span>Fearnley</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/fabio-fognini-sr-competitor-15434/" title="Fabio Fognini" data-id="sr:competitor:15434" data-slug="fabio-fognini-sr-competitor-15434" aria-label="Fabio Fognini"><div class="tc-player"><small class="tc-player__country">ITA</small> <span class="tc-player__name">F. <span>Fognini</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">J. <strong>Fearnley</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">65.5%</span></div></div></div></a></div>"""
    html4 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-08T09:00:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60370637" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60370637" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-08T09:00:00+00:00" data-format="hh:mm">02:00</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-08T09:00:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R128</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60370637/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/hugo-gaston-sr-competitor-223874/" title="Hugo Gaston" data-id="sr:competitor:223874" data-slug="hugo-gaston-sr-competitor-223874" aria-label="Hugo Gaston"><div class="tc-player"><small class="tc-player__country">FRA</small> <span class="tc-player__name">H. <span>Gaston</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/nicolas-jarry-sr-competitor-89632/" title="Nicolas Jarry" data-id="sr:competitor:89632" data-slug="nicolas-jarry-sr-competitor-89632" aria-label="Nicolas Jarry"><div class="tc-player"><small class="tc-player__country">CHI</small> <span class="tc-player__name">N. <span>Jarry</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">N. <strong>Jarry</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">60.2%</span></div></div></div></a></div>"""
    html5 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-08T10:10:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60370625" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60370625" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-08T10:10:00+00:00" data-format="hh:mm">03:10</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-08T10:10:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R128</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60370625/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/fabian-marozsan-sr-competitor-254383/" title="Fabian Marozsan" data-id="sr:competitor:254383" data-slug="fabian-marozsan-sr-competitor-254383" aria-label="Fabian Marozsan"><div class="tc-player"><small class="tc-player__country">HUN</small> <span class="tc-player__name">F. <span>Marozsan</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/joao-fonseca-sr-competitor-863319/" title="Joao Fonseca" data-id="sr:competitor:863319" data-slug="joao-fonseca-sr-competitor-863319" aria-label="Joao Fonseca"><div class="tc-player"><small class="tc-player__country">BRA</small> <span class="tc-player__name">J. <span>Fonseca</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">J. <strong>Fonseca</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">68.3%</span></div></div></div></a></div>"""
    html6 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-08T13:20:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60370633" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60370633" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-08T13:20:00+00:00" data-format="hh:mm">06:20</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-08T13:20:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R128</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60370633/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/alexander-bublik-sr-competitor-163480/" title="Alexander Bublik" data-id="sr:competitor:163480" data-slug="alexander-bublik-sr-competitor-163480" aria-label="Alexander Bublik"><div class="tc-player"><small class="tc-player__country">KAZ</small> <span class="tc-player__name">A. <span>Bublik</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/roman-safiullin-sr-competitor-124930/" title="Roman Safiullin" data-id="sr:competitor:124930" data-slug="roman-safiullin-sr-competitor-124930" aria-label="Roman Safiullin"><div class="tc-player"><small class="tc-player__country"></small> <span class="tc-player__name">R. <span>Safiullin</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">A. <strong>Bublik</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">53.6%</span></div></div></div></a></div>""" 
    html7 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-08T11:20:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60401083" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60401083" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-08T11:20:00+00:00" data-format="hh:mm">04:20</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-08T11:20:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R128</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60401083/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/nuno-borges-sr-competitor-125006/" title="Nuno Borges" data-id="sr:competitor:125006" data-slug="nuno-borges-sr-competitor-125006" aria-label="Nuno Borges"><div class="tc-player"><small class="tc-player__country">PRT</small> <span class="tc-player__name">N. <span>Borges</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/thiago-seyboth-wild-sr-competitor-161262/" title="Thiago Seyboth Wild" data-id="sr:competitor:161262" data-slug="thiago-seyboth-wild-sr-competitor-161262" aria-label="Thiago Seyboth Wild"><div class="tc-player"><small class="tc-player__country">BRA</small> <span class="tc-player__name">T. <span>Seyboth Wild</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">N. <strong>Borges</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">57.1%</span></div></div></div></a></div>"""
    html8 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-08T15:20:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60370621" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60370621" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-08T15:20:00+00:00" data-format="hh:mm">08:20</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-08T15:20:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R128</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60370621/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/learner-tien-sr-competitor-891669/" title="Learner Tien" data-id="sr:competitor:891669" data-slug="learner-tien-sr-competitor-891669" aria-label="Learner Tien"><div class="tc-player"><small class="tc-player__country">USA</small> <span class="tc-player__name">L. <span>Tien</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/reilly-opelka-sr-competitor-130400/" title="Reilly Opelka" data-id="sr:competitor:130400" data-slug="reilly-opelka-sr-competitor-130400" aria-label="Reilly Opelka"><div class="tc-player"><small class="tc-player__country">USA</small> <span class="tc-player__name">R. <span>Opelka</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">L. <strong>Tien</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">50.3%</span></div></div></div></a></div>"""
    html9 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-08T09:00:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60401091" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60401091" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-08T09:00:00+00:00" data-format="hh:mm">02:00</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-08T09:00:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R128</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60401091/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/jesper-de-jong-sr-competitor-310974/" title="Jesper De Jong" data-id="sr:competitor:310974" data-slug="jesper-de-jong-sr-competitor-310974" aria-label="Jesper De Jong"><div class="tc-player"><small class="tc-player__country">NED</small> <span class="tc-player__name">J. <span>De Jong</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/alexander-shevchenko-sr-competitor-371436/" title="Alexander Shevchenko" data-id="sr:competitor:371436" data-slug="alexander-shevchenko-sr-competitor-371436" aria-label="Alexander Shevchenko"><div class="tc-player"><small class="tc-player__country">KAZ</small> <span class="tc-player__name">A. <span>Shevchenko</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">J. <strong>De Jong</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">60.4%</span></div></div></div></a></div>"""
    html10 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-08T13:20:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60370631" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60370631" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-08T13:20:00+00:00" data-format="hh:mm">06:20</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-08T13:20:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R128</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60370631/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/gabriel-diallo-sr-competitor-418213/" title="Gabriel Diallo" data-id="sr:competitor:418213" data-slug="gabriel-diallo-sr-competitor-418213" aria-label="Gabriel Diallo"><div class="tc-player"><small class="tc-player__country">CAN</small> <span class="tc-player__name">G. <span>Diallo</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/marcos-giron-sr-competitor-42379/" title="Marcos Giron" data-id="sr:competitor:42379" data-slug="marcos-giron-sr-competitor-42379" aria-label="Marcos Giron"><div class="tc-player"><small class="tc-player__country">USA</small> <span class="tc-player__name">M. <span>Giron</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">G. <strong>Diallo</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">56.6%</span></div></div></div></a></div>"""
    # List of HTML blocks for upcoming matches
    upcoming_matches_html = [html1, html2, html3, html4, html5, html6, html7, html8, html9, html10]

    # Iterate through each match and display its info
    for i, match_html in enumerate(upcoming_matches_html):
        soup = BeautifulSoup(match_html, 'html.parser')

        # Extract relevant details
        tournament_title = soup.find('h3', class_='tc-tournament-title').get_text(strip=True)
        match_status = soup.find('span', class_='tc-match__status').get_text(strip=True)
        match_time = soup.find('strong', class_='-highlighted').get_text(strip=True)

        player_names = soup.find_all('span', class_='tc-player__name')
        player_home = player_names[0].get_text(strip=True)
        player_away = player_names[1].get_text(strip=True)

        player_countries = soup.find_all('small', class_='tc-player__country')
        country_home = player_countries[0].get_text(strip=True)
        country_away = player_countries[1].get_text(strip=True)

        # Format player names with country
        player_home_formatted = f"{player_home} ({country_home})"
        player_away_formatted = f"{player_away} ({country_away})"

        # Display match info
        st.subheader(f"Tournament: {tournament_title}")
        st.write(f"**Match Status**: {match_status}")
        st.write(f"**Estimated Start Time**: {match_time} AM")
        st.write(f"**Home Player**: {player_home_formatted}")
        st.write(f"**Away Player**: {player_away_formatted}")
        st.markdown("---")  # Divider line
   
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

    content = {}
    prev_name = ''
    for i in range(len(df)):
        if df['utr'][i] > 13:
            curr_name = df['first_name'][i]+' '+df['last_name'][i]
            if curr_name != prev_name:
                curr_name = df['first_name'][i]+' '+df['last_name'][i]
                content[ df['first_name'][i]+' '+df['last_name'][i]] = 100*((df['utr'][i]/df['utr'][i+1])-1)
                                # df['utr'][i]-df['utr'][i+1], 100*((df['utr'][i]/df['utr'][i+1])-1)])
            prev_name = curr_name
    # df = pd.DataFrame(content, columns=["Name", "Previous UTR", "Current UTR", "UTR Change", "UTR % Change"])
    # df = df.sort_values(by="UTR % Change", ascending=False)
    # names = 
    # st.dataframe(df.head(10))

    # df = df.sort_values(by="UTR % Change", ascending=True)
    # st.dataframe(df.head(10))
    # Step 2: Get top 20 up and down movers
    sorted_changes = sorted(content.items(), key=lambda x: abs(x[1]), reverse=True)
    top_movers = sorted_changes[:20]

    # Step 3: Build frequency dict and color mapping
    frequencies = {name: abs(change) * 100 for name, change in top_movers}

    color_map = {name: ("green" if freq > 0 else "red") for name, freq in top_movers}

    # Step 5: Generate and display word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frequencies)
    wordcloud.recolor(color_func=color_func)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")

    st.markdown("### UTR Movers")
    st.pyplot(fig)
with tabs[3]:
    st.title("ATP Rankings (Top 10)")

    # Function to scrape ATP rankings
    def scrape_rankings():
        url = "https://www.atptour.com/en/rankings/singles?rankRange=0-100"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        rows = soup.select("table.mega-table tbody tr")

        rankings = []
        count = 0
        for row in rows:
            try:
                rank_td = row.find("td", class_="rank")
                player_td = row.find("td", class_="player")
                points_td = row.find("td", class_="points")

                # Skip if any piece is missing
                if not (rank_td and player_td and points_td):
                    continue

                # Rank
                rank = rank_td.get_text(strip=True)

                # Name (full name)
                name_tag = player_td.select_one("li.name a")
                first_name = name_tag.get_text(strip=True).split()[0]  # Assuming first name is first part
                last_name = name_tag.get_text(strip=True).split()[-1]  # Assuming last name is the last part
                full_name = f"{first_name} {last_name}"

                # Country code (optional)
                flag_use = player_td.select_one("li.avatar use")
                country_code = flag_use["href"].split("-")[-1].upper() if flag_use else "N/A"

                # Points
                points = points_td.get_text(strip=True).replace(",", "")

                rankings.append((rank, full_name, country_code, points))
                count += 1

                if count == 10:
                    break

            except Exception as e:
                st.error(f"Error parsing row: {e}")

        return rankings

    rankings = scrape_rankings()

    if rankings:
        # Display the rankings in a properly formatted table
        st.write("### Top 10 ATP Rankings")
        st.markdown("| Rank | Name | Country | Points |")
        st.markdown("|------|------|---------|--------|")
        for rank, name, country, points in rankings:
            st.markdown(f"| {rank} | {name} | {country} | {points} |")
    else:
        st.write("Failed to retrieve rankings.")

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

    # def collect_feedback():
    # Create a form to collect feedback
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

    # collect_feedback()
