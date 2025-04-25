import numpy as np
import random
import pandas as pd
import torch
import joblib
from colorama import Fore, Style, init
import torch
import torch.nn as nn
from st_files_connection import FilesConnection
import streamlit as st
import joblib
from google.cloud import storage
import io
from google.oauth2 import service_account
import matplotlib.pyplot as plt

# GCS Buckets and files
MODEL_BUCKET = "utr-model-training-bucket"
MODEL_BLOB = "model.sav"
UTR_BUCKET = "utr_scraper_bucket"
UTR_FILE = "utr_history.csv"
MATCHES_BUCKET = "matches-scraper-bucket"
MATCHES_FILE = "atp_utr_tennis_matches.csv"

# Tennis Predictor Model
class TennisPredictor(nn.Module):
    def __init__(self, input_size):
        super(TennisPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 1028)
        self.fc2 = nn.Linear(1028, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.sigmoid(self.fc8(x))
        return x


class MarkovModel:
    def __init__(self, prop):
        self.curr_state = '0-0'
        self.prop = prop
        if round(0.66*(0.5+prop),10)+round(1-(0.66*(0.5+prop)),10) == 1:
            self.pprop = round(0.66*(0.5+prop),10)
            self.inv_prop = round(1-(0.66*(0.5+prop)),10)
        else:
            self.pprop = round(0.66*(0.5+prop),10) + 0.0000000001
            self.inv_prop = round(1-(0.66*(0.5+prop)),10)
        self.deuce = round((self.pprop**2) / (1 - 2*self.pprop*self.inv_prop),10)
        self.inv_deuce = round(1-((self.pprop**2) / (1 - (2*self.pprop*self.inv_prop))),10)

        self.pt_matrix = {
            '0-0': {'0-0': 0, '0-15': self.inv_prop, '15-0': self.pprop, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': 0, '30-15': 0, '0-40': 0, '40-0': 0, '15-40': 0, '40-15': 0, '30-30(DEUCE)': 0, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': 0, 'BREAK': 0},
            '0-15': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': self.pprop, '0-30': self.inv_prop, '30-0': 0, '15-30': 0, '30-15': 0, '0-40': 0, '40-0': 0, '15-40': 0, '40-15': 0, '30-30(DEUCE)': 0, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': 0, 'BREAK': 0},
            '15-0': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': self.inv_prop, '0-30': 0, '30-0': self.pprop, '15-30': 0, '30-15': 0, '0-40': 0, '40-0': 0, '15-40': 0, '40-15': 0, '30-30(DEUCE)': 0, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': 0, 'BREAK': 0},
            '15-15': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': self.inv_prop, '30-15': self.pprop, '0-40': 0, '40-0': 0, '15-40': 0, '40-15': 0, '30-30(DEUCE)': 0, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': 0, 'BREAK': 0},
            '0-30': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': self.pprop, '30-15': 0, '0-40': self.inv_prop, '40-0': 0, '15-40': 0, '40-15': 0, '30-30(DEUCE)': 0, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': 0, 'BREAK': 0},
            '30-0': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': 0, '30-15': self.inv_prop, '0-40': 0, '40-0': self.pprop, '15-40': 0, '40-15': 0, '30-30(DEUCE)': 0, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': 0, 'BREAK': 0},
            '15-30': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': 0, '30-15': 0, '0-40': 0, '40-0': 0, '15-40': self.inv_prop, '40-15': 0, '30-30(DEUCE)': self.pprop, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': 0, 'BREAK': 0},
            '30-15': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': 0, '30-15': 0, '0-40': 0, '40-0': 0, '15-40': 0, '40-15': self.pprop, '30-30(DEUCE)': self.inv_prop, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': 0, 'BREAK': 0},
            '0-40': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': 0, '30-15': 0, '0-40': 0, '40-0': 0, '15-40': self.pprop, '40-15': 0, '30-30(DEUCE)': 0, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': 0, 'BREAK': self.inv_prop},
            '40-0': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': 0, '30-15': 0, '0-40': 0, '40-0': 0, '15-40': 0, '40-15': self.inv_prop, '30-30(DEUCE)': 0, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': self.pprop, 'BREAK': 0},
            '15-40': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': 0, '30-15': 0, '0-40': 0, '40-0': 0, '15-40': 0, '40-15': 0, '30-30(DEUCE)': 0, '30-40(40-A)': self.pprop, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': 0, 'BREAK': self.inv_prop},
            '40-15': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': 0, '30-15': 0, '0-40': 0, '40-0': 0, '15-40': 0, '40-15': 0, '30-30(DEUCE)': 0, '30-40(40-A)': 0, '40-30(A-40)': self.inv_prop, '40-40(NO AD)': 0, 'HOLD': self.pprop, 'BREAK': 0},
            '30-30(DEUCE)': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': 0, '30-15': 0, '0-40': 0, '40-0': 0, '15-40': 0, '40-15': 0, '30-30(DEUCE)': 0, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': self.deuce, 'BREAK': self.inv_deuce},
            '30-40(40-A)': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': 0, '30-15': 0, '0-40': 0, '40-0': 0, '15-40': 0, '40-15': 0, '30-30(DEUCE)': self.pprop, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': 0, 'BREAK': self.inv_prop},
            '40-30(A-40)': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': 0, '30-15': 0, '0-40': 0, '40-0': 0, '15-40': 0, '40-15': 0, '30-30(DEUCE)': self.inv_prop, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': self.pprop, 'BREAK': 0},
            '40-40(NO AD)': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': 0, '30-15': 0, '0-40': 0, '40-0': 0, '15-40': 0, '40-15': 0, '30-30(DEUCE)': 0, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': self.pprop, 'BREAK': self.inv_prop},
            'HOLD': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': 0, '30-15': 0, '0-40': 0, '40-0': 0, '15-40': 0, '40-15': 0, '30-30(DEUCE)': 0, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': 1.0, 'BREAK': 0},
            'BREAK': {'0-0': 0, '0-15': 0, '15-0': 0, '15-15': 0, '0-30': 0, '30-0': 0, '15-30': 0, '30-15': 0, '0-40': 0, '40-0': 0, '15-40': 0, '40-15': 0, '30-30(DEUCE)': 0, '30-40(40-A)': 0, '40-30(A-40)': 0, '40-40(NO AD)': 0, 'HOLD': 0, 'BREAK': 1.0},
        }
    
    def next_state(self):
        try:
            self.curr_state = np.random.choice(list(self.pt_matrix.keys()), p=list(self.pt_matrix[self.curr_state].values()))
        except:
            print(list(self.pt_matrix[self.curr_state].values()))
            quit()
        return self.curr_state


def game(prop):
    model = MarkovModel(prop)
    while model.curr_state != 'HOLD' and model.curr_state != 'BREAK':
        model.next_state()
    return model.curr_state


def create_score(prop, best_of):
    score = ''
    first_serve = random.randint(0,1)
    sets_won = 0
    num_sets = 0
    for _ in range(best_of):
        p1_games = 0
        p2_games = 0
        done = True
        while done:
            if p1_games == 6 and p2_games < 5 or p2_games == 6 and p1_games < 5: # Good
                break
            elif p1_games == 7 or p2_games == 7:
                break
            
            if (p1_games+p2_games) % 2 == 0: # Good
                hb = game(prop)
            else:
                hb = game(1-prop)

            if first_serve == 0: # Good
                if hb == 'HOLD' and (p1_games+p2_games) % 2 == 0:
                    p1_games += 1
                elif hb == 'HOLD' and (p1_games+p2_games) % 2 == 1:
                    p2_games += 1
                elif hb == 'BREAK' and (p1_games+p2_games) % 2 == 0:
                    p2_games += 1
                elif hb == 'BREAK' and (p1_games+p2_games) % 2 == 1:
                    p1_games += 1
            else:
                if hb == 'HOLD' and (p1_games+p2_games) % 2 == 0:
                    p2_games += 1
                elif hb == 'HOLD' and (p1_games+p2_games) % 2 == 1:
                    p1_games += 1
                elif hb == 'BREAK' and (p1_games+p2_games) % 2 == 0:
                    p1_games += 1
                elif hb == 'BREAK' and (p1_games+p2_games) % 2 == 1:
                    p2_games += 1

        num_sets += 1 # Good
        if p1_games > p2_games:
            sets_won += 1
        else:
            sets_won -= 1
        score = score + str(p1_games) + '-' + str(p2_games) + ' '
        if abs(sets_won) == round(best_of/3)+1:
            break
        elif abs(sets_won) == 2 and num_sets > 2:
            break
    score = score[:-1]
    # print(score)
    return score


def preprocess_player_data(p1, p2, profiles):
    match_vector = [profiles[p1]['utr']-profiles[p2]['utr'], 
                    profiles[p1]['win_vs_lower'],
                    profiles[p2]['win_vs_lower'],
                    profiles[p1]['win_vs_higher'],
                    profiles[p2]['win_vs_higher'],
                    profiles[p1]['recent10'],
                    profiles[p2]['recent10'],
                    profiles[p1]['wvl_utr'],
                    profiles[p2]['wvl_utr'],
                    profiles[p1]['wvh_utr'],
                    profiles[p2]['wvh_utr'],
                    profiles[p1]['h2h'][p2][0] / profiles[p1]['h2h'][p2][1],
                    profiles[p2]['h2h'][p1][0] / profiles[p2]['h2h'][p1][1]
                    ]
    return match_vector


def get_prop(model, p1, p2, player_profiles):
    # Make one prediction
    X = preprocess_player_data(p1, p2, player_profiles)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    prop = model(X_tensor).squeeze().detach().numpy()
    prop = 1-float(prop)
    return prop


def find_winner(score):
    p1_sets_won = 0
    p2_sets_won = 0
    for j in range(len(score)):
        if j % 4 == 0:
            if int(score[j]) > int(score[j+2]):
                p1_sets_won += 1
            else:
                p2_sets_won += 1
    if p1_sets_won > p2_sets_won:
        pred_winner = 'p1'
    else:
        pred_winner = 'p2'
    return pred_winner


# def make_prediction(p1, p2, location, best_of=3):
#     conn = st.connection('gcs', type=FilesConnection)
#     data = conn.read("matches-scraper-bucket/atp_utr_tennis_matches.csv", input_format="csv", ttl=600)
#     utr_history = conn.read("utr_scraper_bucket/utr_history.csv", input_format="csv", ttl=600)

#     model = joblib.load('model.sav')

#     history = get_player_history(utr_history)
#     player_profiles = get_player_profiles(data, history)

#     prop = get_prop(model, p1, p2, player_profiles)
#     score = create_score(prop, best_of)

#     pred_winner = find_winner(score)
#     if prop >= 0.5:
#         true_winner = 'p1'
#     else:
#         true_winner = 'p2'

#     while true_winner != pred_winner:
#         score = create_score(prop, best_of)
#         pred_winner = find_winner(score)

#     prediction = ""

#     if true_winner == 'p1':
#         prediction += f'{p1} is predicted to win against {p2} ({round(100*prop, 2)}% Probability): '
#     else:
#         prediction += f'{p1} is predicted to lose against {p2} ({round(100*(1-prop), 2)}% Probability): '
#     for i in range(len(score)):
#         if i % 4 == 0 and int(score[i]) > int(score[i+2]):
#             prediction += score[i]
#         elif i % 4 == 0 and int(score[i]) < int(score[i+2]):
#             prediction += score[i]
#         else:
#             prediction += score[i]

#     return prediction


def get_set_player_profiles(matches, history, st=None):
    """Same logic you used in training (simplified to what preprocess needs)."""
    profiles = {}
    # st.write("Gathering Profile Data")
    for r in matches.itertuples():
        for plyr, opp in ((r.p1, r.p2), (r.p2, r.p1)):
            if plyr not in profiles:
                profiles[plyr] = {
                    "win_vs_lower": [], "win_vs_higher": [],
                    "recent10": [], "wvl_utr": [], "wvh_utr": [],
                    "h2h": {}, "utr": history.get(plyr, getattr(r, "p1_utr" if plyr == r.p1 else "p2_utr"))
                }
            if opp not in profiles[plyr]["h2h"]:
                profiles[plyr]["h2h"][opp] = [0, 0]  # Wins, Total
                
            # minimal updates just so preprocess() works
            
    # st.markdown("Gather Profile Data")
    for r in matches.itertuples():
        # for plyr, opp in ((r.p1, r.p2), (r.p2, r.p1)):
            if r.winner == r.p1:
                profiles[r.p1]['h2h'][r.p2][0] += 1
                profiles[r.p1]['h2h'][r.p2][1] += 1
                profiles[r.p2]['h2h'][r.p1][1] += 1
            else:
                profiles[r.p1]['h2h'][r.p2][1] += 1
                profiles[r.p2]['h2h'][r.p1][0] += 1
                profiles[r.p2]['h2h'][r.p1][1] += 1
            
            # Record win rates vs higher/lower-rated opponents
            if r.p1_utr-r.p2_utr > 0:  # Player faced a lower-rated opponent
                if r.winner == r.p1:
                    profiles[r.p1]["win_vs_lower"].append(1)
                    profiles[r.p2]["win_vs_higher"].append(0)
                    profiles[r.p1]["wvl_utr"].append(r.p2_utr)
                    profiles[r.p2]["wvh_utr"].append(0)
                else:
                    profiles[r.p1]["win_vs_lower"].append(0)
                    profiles[r.p2]["win_vs_higher"].append(1)
                    profiles[r.p2]["wvh_utr"].append(r.p1_utr)
                    profiles[r.p1]["wvl_utr"].append(0)

            else:  # Player faced a higher-rated opponent
                if r.winner == r.p1:
                    profiles[r.p1]["win_vs_higher"].append(1)
                    profiles[r.p2]["win_vs_lower"].append(0)
                    profiles[r.p1]["wvh_utr"].append(r.p2_utr)
                    profiles[r.p2]["wvl_utr"].append(0)
                else:
                    profiles[r.p1]["win_vs_higher"].append(0)
                    profiles[r.p2]["win_vs_lower"].append(1)
                    profiles[r.p2]["wvl_utr"].append(r.p1_utr)
                    profiles[r.p1]["wvh_utr"].append(0)

            if r.winner == r.p1:
                profiles[r.p1]["recent10"].append(1)
                profiles[r.p2]["recent10"].append(0)
            else:
                profiles[r.p1]["recent10"].append(0)
                profiles[r.p2]["recent10"].append(1)

            if len(profiles[r.p1]["recent10"]) > 10:
                profiles[r.p1]["recent10"] = profiles[r.p1]["recent10"][1:]
            if len(profiles[r.p2]["recent10"]) > 10:
                profiles[r.p2]["recent10"] = profiles[r.p2]["recent10"][1:]
            
    # convert lists → means so they’re scalar
    for p in profiles.values():
        for k in ("win_vs_lower", "win_vs_higher", "recent10", "wvl_utr", "wvh_utr"):
            p[k] = np.mean(p[k]) if p[k] else 0
    return profiles


def get_set_player_profiles_general(data, history):
    player_profiles = {}

    for i in range(len(data)):
        player = data['p1'][i]
        opponent = data['p2'][i]
        utr_diff = data['p1_utr'][i] - data['p2_utr'][i]
        
        if player not in player_profiles and player in history:
            player_profiles[player] = {
                "win_vs_lower": [],
                "wvl_utr": [],
                "win_vs_higher": [],
                "wvh_utr": [],
                "recent10": [],
                "r10_utr": [],
                "utr": history[player]['utr'],
                "h2h": {}
            }
        elif player not in player_profiles:
            player_profiles[player] = {
                "win_vs_lower": [],
                "wvl_utr": [],
                "win_vs_higher": [],
                "wvh_utr": [],
                "recent10": [],
                "r10_utr": [],
                "utr": data['p1_utr'][i] if data['p1'][i] == player else data['p2_utr'][i],
                "h2h": {}
            }

        if opponent not in player_profiles and opponent in history:
            player_profiles[opponent] = {
                "win_vs_lower": [],
                "wvl_utr": [],
                "win_vs_higher": [],
                "wvh_utr": [],
                "recent10": [],
                "r10_utr": [],
                "utr": history[opponent]['utr'][0],
                "h2h": {}
            }
        elif opponent not in player_profiles:
            player_profiles[opponent] = {
                "win_vs_lower": [],
                "wvl_utr": [],
                "win_vs_higher": [],
                "wvh_utr": [],
                "recent10": [],
                "r10_utr": [],
                "utr": data['p1_utr'][i] if data['p1'][i] == opponent else data['p2_utr'][i],
                "h2h": {}
            }

        if opponent not in player_profiles[player]['h2h']:
            player_profiles[player]['h2h'][opponent] = [0,0]

        if player not in player_profiles[opponent]['h2h']:
            player_profiles[opponent]['h2h'][player] = [0,0]

        if data['winner'][i] == player:
            player_profiles[player]['h2h'][opponent][0] += 1
            player_profiles[player]['h2h'][opponent][1] += 1
            player_profiles[opponent]['h2h'][player][1] += 1
        else:
            player_profiles[player]['h2h'][opponent][1] += 1
            player_profiles[opponent]['h2h'][player][0] += 1
            player_profiles[opponent]['h2h'][player][1] += 1
        
        # Record win rates vs higher/lower-rated opponents
        if utr_diff > 0:  # Player faced a lower-rated opponent
            if data["winner"][i] == player:
                player_profiles[player]["win_vs_lower"].append(1)
                player_profiles[opponent]["win_vs_higher"].append(0)
                player_profiles[player]["wvl_utr"].append(data["p2_utr"][i])
                player_profiles[opponent]["wvh_utr"].append(0)
            else:
                player_profiles[player]["win_vs_lower"].append(0)
                player_profiles[opponent]["win_vs_higher"].append(1)
                player_profiles[opponent]["wvh_utr"].append(data["p1_utr"][i])
                player_profiles[player]["wvl_utr"].append(0)

        else:  # Player faced a higher-rated opponent
            if data["winner"][i] == player:
                player_profiles[player]["win_vs_higher"].append(1)
                player_profiles[opponent]["win_vs_lower"].append(0)
                player_profiles[player]["wvh_utr"].append(data["p2_utr"][i])
                player_profiles[opponent]["wvl_utr"].append(0)
            else:
                player_profiles[player]["win_vs_higher"].append(0)
                player_profiles[opponent]["win_vs_lower"].append(1)
                player_profiles[opponent]["wvl_utr"].append(data["p1_utr"][i])
                player_profiles[player]["wvh_utr"].append(0)

        if data['winner'][i] == player:
            player_profiles[player]["recent10"].append(1)
            player_profiles[opponent]["recent10"].append(0)
        else:
            player_profiles[player]["recent10"].append(0)
            player_profiles[opponent]["recent10"].append(1)

        if len(player_profiles[player]["recent10"]) > 10:
            player_profiles[player]["recent10"] = player_profiles[player]["recent10"][1:]
        if len(player_profiles[opponent]["recent10"]) > 10:
            player_profiles[opponent]["recent10"] = player_profiles[opponent]["recent10"][1:]

    for player in player_profiles:
        profile = player_profiles[player]
        profile["win_vs_lower"] = np.mean(profile["win_vs_lower"]) if len(profile["win_vs_lower"]) > 0 else 0
        profile["win_vs_higher"] = np.mean(profile["win_vs_higher"]) if len(profile["win_vs_higher"]) > 0 else 0
        profile["recent10"] = np.mean(profile["recent10"]) if len(profile["recent10"]) > 0 else 0
        profile['wvl_utr'] = np.mean(profile['wvl_utr']) if len(profile['wvl_utr']) > 0 else 0
        profile['wvh_utr'] = np.mean(profile['wvh_utr']) if len(profile['wvh_utr']) > 0 else 0
    return player_profiles


def get_player_history(df):
    """Latest UTR for each player."""
    out = {}
    for row in df.itertuples():
        key = f"{row.first_name} {row.last_name}"
        out.setdefault(key, []).append((row.utr, row.date))

    result = {}
    for player, utr_list in out.items():
        # Sort the list of (utr, date) tuples by the date (index 1)
        utr_list.sort(key=lambda tup: tup[1])  # lambda here is safe and readable
        result[player] = utr_list[-1][0]  # Most recent UTR

    return result


def get_player_history_general(utr_history):
    history = {}

    for i in range(len(utr_history)):
        if utr_history['first_name'][i]+' '+utr_history['last_name'][i] not in history:
            history[utr_history['first_name'][i]+' '+utr_history['last_name'][i]] = {
                'utr': [utr_history['utr'][i]],
                'date': [utr_history['date'][i]]
            }
        else:
            history[utr_history['first_name'][i]+' '+utr_history['last_name'][i]]['utr'].append(utr_history['utr'][i])
            history[utr_history['first_name'][i]+' '+utr_history['last_name'][i]]['date'].append(utr_history['date'][i])

    return history


def download_model_from_gcs(credentials_dict, bucket_name, source_blob_name, destination_file_name):
    """Download a joblib model file from a GCS bucket to the local filesystem."""
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    client = storage.Client(credentials=credentials, project=credentials_dict["project_id"])
    
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    # print(f"Downloaded {source_blob_name} to {destination_file_name}")


def load_model(credentials_dict):
    bucket_name = "utr-model-training-bucket"              
    source_blob_name = "model.sav"            
    local_path = "/model.sav"      
               
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    client = storage.Client(credentials=credentials, project=credentials_dict["project_id"])
    
    download_model_from_gcs(credentials_dict, bucket_name, source_blob_name, local_path)
    model = joblib.load(local_path)
    return model


def make_prediction(credentials_dict, p1, p2, location, best_of=3):
    conn = st.connection('gcs', type=FilesConnection)
    data = conn.read("matches-scraper-bucket/atp_utr_tennis_matches.csv", input_format="csv", ttl=600)
    utr_history = conn.read("utr_scraper_bucket/utr_history.csv", input_format="csv", ttl=600)

    # model = joblib.load('model.sav')

    model = load_model(credentials_dict)

    history = get_player_history(utr_history)
    player_profiles = get_player_profiles(data, history, p1, p2)

    prop = get_prop(model, p1, p2, player_profiles)
    score = create_score(prop, best_of)

    pred_winner = find_winner(score)
    if prop >= 0.5:
        true_winner = 'p1'
    else:
        true_winner = 'p2'

    while true_winner != pred_winner:
        score = create_score(prop, best_of)
        pred_winner = find_winner(score)

    prediction = ""

    if true_winner == 'p1':
        prediction += f'{p1} is predicted to win against {p2} ({round(100*prop, 2)}% Probability): '
    else:
        prediction += f'{p1} is predicted to lose against {p2} ({round(100*(1-prop), 2)}% Probability): '
    for i in range(len(score)):
        if i % 4 == 0 and int(score[i]) > int(score[i+2]):
            prediction += score[i]
        elif i % 4 == 0 and int(score[i]) < int(score[i+2]):
            prediction += score[i]
        else:
            prediction += score[i]

    return prediction


def download_csv_from_gcs(credentials_dict, bucket, file_path):
    """Downloads a CSV from GCS and returns a pandas DataFrame."""
    
    try:
        # use global credentials dict
        # Initialize client (credentials are picked up from st.secrets)
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        
        # Initialize the GCS client with credentials and project
        client = storage.Client(credentials=credentials, project=credentials_dict["project_id"])
        
        blob = bucket.blob(file_path)
        data = blob.download_as_string()
        df = pd.read_csv(io.BytesIO(data))
        return df
    except Exception as e:
        raise



# Rewritten preprocess_player_data function, handles division by zero error
def preprocess_match_data(match_row, profiles):
    """
    Build the feature vector for one match.

    Parameters
    ----------
    match_row : dict-like (e.g. a pandas Series or plain dict)
        Must contain keys 'p1', 'p2', 'p1_utr', 'p2_utr'.
    profiles : dict
        Nested profile data exactly as in your original script.

    Returns
    -------
    list[float]
        Feature vector identical to the one returned by preprocess_match_data.
    """
    p1 = match_row["p1"]
    p2 = match_row["p2"]

    # Helper that returns win-ratio with a safe fallback (0 wins / 1 match)
    def h2h_ratio(player, opponent):
        h2h_stats = profiles[player]["h2h"].get(opponent, [0, 1])
        wins = h2h_stats[0]
        total = h2h_stats[1] if h2h_stats[1] != 0 else 1  # prevent division by zero
        return wins / total

    vec = [
        match_row["p1_utr"] - match_row["p2_utr"],
        profiles[p1]["win_vs_lower"],  profiles[p2]["win_vs_lower"],
        profiles[p1]["win_vs_higher"], profiles[p2]["win_vs_higher"],
        profiles[p1]["recent10"],      profiles[p2]["recent10"],
        profiles[p1]["wvl_utr"],       profiles[p2]["wvl_utr"],
        profiles[p1]["wvh_utr"],       profiles[p2]["wvh_utr"],
        h2h_ratio(p1, p2),
        h2h_ratio(p2, p1),
    ]
    return vec

def display_player_metrics(player1, player2, history, profiles):
    if player1 != "" and player2 != "":
        profile = profiles[player1]
        # st.markdown(profile)

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
    if player1 != "" and player2 != "":
        # st.markdown(history[player1])
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