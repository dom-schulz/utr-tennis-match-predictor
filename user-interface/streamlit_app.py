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

    html1 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-09T11:00:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60272137" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60272137" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-09T11:00:00+00:00" data-format="hh:mm">04:00</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-09T11:00:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R64</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60272137/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/dusan-lajovic-sr-competitor-39234/" title="Dusan Lajovic" data-id="sr:competitor:39234" data-slug="dusan-lajovic-sr-competitor-39234" aria-label="Dusan Lajovic"><div class="tc-player"><small class="tc-player__country">SRB</small> <span class="tc-player__name">D. <span>Lajovic</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/carlos-alcaraz-garfia-sr-competitor-407573/" title="Carlos Alcaraz" data-id="sr:competitor:407573" data-slug="carlos-alcaraz-garfia-sr-competitor-407573" aria-label="Carlos Alcaraz"><div class="tc-player"><small class="tc-player__country">ESP</small> <span class="tc-player__name">C. <span>Alcaraz</span></span> <small class="tc-player__seeding">(3)</small></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">C. <strong>Alcaraz</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">89.9%</span></div></div></div></a></div>"""
    html2 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-09T11:00:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60272093" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60272093" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-09T11:00:00+00:00" data-format="hh:mm">04:00</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-09T11:00:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R64</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60272093/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/alexandre-muller-sr-competitor-88992/" title="Alexandre Muller" data-id="sr:competitor:88992" data-slug="alexandre-muller-sr-competitor-88992" aria-label="Alexandre Muller"><div class="tc-player"><small class="tc-player__country">FRA</small> <span class="tc-player__name">A. <span>Muller</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/stefanos-tsitsipas-sr-competitor-122366/" title="Stefanos Tsitsipas" data-id="sr:competitor:122366" data-slug="stefanos-tsitsipas-sr-competitor-122366" aria-label="Stefanos Tsitsipas"><div class="tc-player"><small class="tc-player__country">GRE</small> <span class="tc-player__name">S. <span>Tsitsipas</span></span> <small class="tc-player__seeding">(18)</small></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">S. <strong>Tsitsipas</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">76.2%</span></div></div></div></a></div>"""
    html3 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-09T12:10:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60272081" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60272081" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-09T12:10:00+00:00" data-format="hh:mm">05:10</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-09T12:10:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R64</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60272081/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/cameron-norrie-sr-competitor-95935/" title="Cameron Norrie" data-id="sr:competitor:95935" data-slug="cameron-norrie-sr-competitor-95935" aria-label="Cameron Norrie"><div class="tc-player"><small class="tc-player__country">GBR</small> <span class="tc-player__name">C. <span>Norrie</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/daniil-medvedev-sr-competitor-163504/" title="Daniil Medvedev" data-id="sr:competitor:163504" data-slug="daniil-medvedev-sr-competitor-163504" aria-label="Daniil Medvedev"><div class="tc-player"><small class="tc-player__country"></small> <span class="tc-player__name">D. <span>Medvedev</span></span> <small class="tc-player__seeding">(10)</small></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">D. <strong>Medvedev</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">72.1%</span></div></div></div></a></div>"""
    html4 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-09T14:30:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60272131" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60272131" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-09T14:30:00+00:00" data-format="hh:mm">07:30</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-09T14:30:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R64</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60272131/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/jack-draper-sr-competitor-352776/" title="Jack Draper" data-id="sr:competitor:352776" data-slug="jack-draper-sr-competitor-352776" aria-label="Jack Draper"><div class="tc-player"><small class="tc-player__country">GBR</small> <span class="tc-player__name">J. <span>Draper</span></span> <small class="tc-player__seeding">(5)</small></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/luciano-darderi-sr-competitor-534043/" title="Luciano Darderi" data-id="sr:competitor:534043" data-slug="luciano-darderi-sr-competitor-534043" aria-label="Luciano Darderi"><div class="tc-player"><small class="tc-player__country">ITA</small> <span class="tc-player__name">L. <span>Darderi</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">J. <strong>Draper</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">75.9%</span></div></div></div></a></div>"""
    html5 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-09T09:00:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60272091" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60272091" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-09T09:00:00+00:00" data-format="hh:mm">02:00</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-09T09:00:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R64</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60272091/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/arthur-fils-sr-competitor-671637/" title="Arthur Fils" data-id="sr:competitor:671637" data-slug="arthur-fils-sr-competitor-671637" aria-label="Arthur Fils"><div class="tc-player"><small class="tc-player__country">FRA</small> <span class="tc-player__name">A. <span>Fils</span></span> <small class="tc-player__seeding">(13)</small></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/tallon-griekspoor-sr-competitor-122368/" title="Tallon Griekspoor" data-id="sr:competitor:122368" data-slug="tallon-griekspoor-sr-competitor-122368" aria-label="Tallon Griekspoor"><div class="tc-player"><small class="tc-player__country">NED</small> <span class="tc-player__name">T. <span>Griekspoor</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">A. <strong>Fils</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">63.7%</span></div></div></div></a></div>"""
    html6 = """<div class="fm-card tc-match -smaller" data-start-time="2025-05-09T09:00:00+00:00" data-match-status="prematch" data-match-slug="sr-match-60272085" data-tournament-slug="sr-tournament-2779-rome-italy" id="live-update-sr-match-60272085" data-event="Internazionali BNL d'Italia"><div class="tc-match__header"><div class="tc-match__header-top"><h3 class="tc-tournament-title"><a href="/tournaments/sr-tournament-2779-rome-italy/" class="tc-tournament-title-link" title="Internazionali BNL d'Italia">Internazionali BNL d'Italia</a></h3></div><div class="tc-match__header-bottom"><div class="tc-match__header-left"><span class="tc-match__status" js-match-card-status="">Not Started</span><div class="tc-match__cta" js-match-card-buttons=""></div><div class="tc-time" js-match-card-start-time=""><div class="tc-time__label"><span class="tc-time__label__text">Estimated Start</span></div><div class="tc-time__hour"><strong class="-highlighted" js-local-time="" data-utc-time="2025-05-09T09:00:00+00:00" data-format="hh:mm">02:00</strong> <span class="tc-time__hour--smaller" js-local-time="" data-utc-time="2025-05-09T09:00:00+00:00" data-format="A">AM</span></div></div></div><div class="tc-match__header-right"><div class="tc-match__info"><span class="tc-round-name">R64</span> <span class="mx-01">-</span> <span class="tc-event-title">Men's Singles</span></div></div></div></div><a href="/tournaments/sr-tournament-2779-rome-italy/sr-match-60272085/" class="tc-match__content-outer"><div class="tc-match__content"><div class="tc-match__items"><div class="tc-match__item -home" js-match-card-home-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/jordan-thompson-sr-competitor-87690/" title="Jordan Thompson" data-id="sr:competitor:87690" data-slug="jordan-thompson-sr-competitor-87690" aria-label="Jordan Thompson"><div class="tc-player"><small class="tc-player__country">AUS</small> <span class="tc-player__name">J. <span>Thompson</span></span></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div><div class="tc-match__item -away" js-match-card-away-player=""><div class="tc-player"><div class="tc-player--wrap"><div class="tc-player--wrap--inner"><object><a class="tc-player__link" href="/players-rankings/brandon-nakashima-sr-competitor-294300/" title="Brandon Nakashima" data-id="sr:competitor:294300" data-slug="brandon-nakashima-sr-competitor-294300" aria-label="Brandon Nakashima"><div class="tc-player"><small class="tc-player__country">USA</small> <span class="tc-player__name">B. <span>Nakashima</span></span> <small class="tc-player__seeding">(28)</small></div></a></object></div></div></div><div class="tc-match__stats--wrap" js-match-card-score-container=""><div><small>&nbsp;</small></div></div></div></div><div class="tc-prediction" js-match-card-predictions=""><strong class="tc-prediction__title">Win Probability</strong> <span class="tc-prediction__name">B. <strong>Nakashima</strong></span><div class="tc-prediction__box"><span class="tc-prediction__value">65.7%</span></div></div></div></a></div>"""
    # List of HTML blocks for upcoming matches
    upcoming_matches_html = [html1, html2, html3, html4, html5, html6]

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

        # <div class="tc-player-seeding">Seeded:&nbsp;3</div>
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

# --- ATP Rankings Tab ---
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
        # Display the rankings with spacing, but without boxes
        st.markdown("""
            <div style="font-family: Arial, sans-serif;">
                <h3 style="text-align: left; color: #212529; margin-bottom: 20px; font-size: 1.5rem; font-weight: bold;">Top 10 ATP Rankings</h3>
                <div style="display: flex;  align-items: center; padding: 8px 0; border-bottom: 1px solid #ddd;">
                    <span style="width: 10%; text-align: left; color: #212529; font-weight: bold;">Rank</span>
                    <span style="width: 40%; text-align: left; color: #212529; font-weight: bold;">Name</span>
                    <span style="width: 25%; text-align: center; color: #212529; font-weight: bold;">Country</span>
                    <span style="width: 25%; text-align: right; color: #212529; font-weight: bold;">Points</span>
                </div>
        """, unsafe_allow_html=True)

        for rank, name, country, points in rankings:
            st.markdown(f"""
                <div style="display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid #eee;">
                    <span style="width: 10%; text-align: left;">{rank}</span>
                    <span style="width: 40%; text-align: left; font-weight: 400;">{name}</span>
                    <span style="width: 25%; text-align: center;">{country}</span>
                    <span style="width: 25%; text-align: right; font-weight: 400;">{points}</span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

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
