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

def predict(p1, p2, location, best_of=3):
    conn = st.connection('gcs', type=FilesConnection)
    data = conn.read("matches-scraper-bucket/atp_utr_tennis_matches.csv", input_format="csv", ttl=600)
    utr_history = conn.read("utr_scraper_bucket/utr_history.csv", input_format="csv", ttl=600)

    model = joblib.load('model.sav')

    history = get_player_history(utr_history)
    player_profiles = get_player_profiles(data, history)

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

def get_player_profiles(data, history, p1, p2):
    player_profiles = {}

    for i in range(len(data)):
        for player, opponent in [(data['p1'][i], data['p2'][i]), (data['p2'][i], data['p1'][i])]:
            if player == p1 or player == p2:
                utr_diff = data['p1_utr'][i] - data['p2_utr'][i] if data['p1'][i] == player else data['p2_utr'][i] - data['p1_utr'][i]
                
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
                        "utr": history[opponent]['utr'],
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
                    player_profiles[player]['h2h'][opponent] = [0,0,1,1]

                if player not in player_profiles[opponent]['h2h']:
                    player_profiles[opponent]['h2h'][player] = [0,0,1,1]

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
                    player_profiles[player]["win_vs_lower"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)
                    player_profiles[player]["wvl_utr"].append(data["p2_utr"][i] if data["p1"][i] == player else data["p1_utr"][i])

                else:  # Player faced a higher-rated opponent
                    player_profiles[player]["win_vs_higher"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)
                    player_profiles[player]["wvh_utr"].append(data["p2_utr"][i] if data["p1"][i] == player else data["p1_utr"][i])

                if len(player_profiles[player]["recent10"]) < 10:
                    player_profiles[player]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)
                else:
                    player_profiles[player]["recent10"] = player_profiles[player]["recent10"][1:]
                    player_profiles[player]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)

                if len(player_profiles[opponent]["recent10"]) < 10:
                    player_profiles[opponent]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == opponent else data["p_win"][i] == 0)
                else:
                    player_profiles[opponent]["recent10"] = player_profiles[opponent]["recent10"][1:]
                    player_profiles[opponent]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == opponent else data["p_win"][i] == 0)
        # except:
        #     continue # player/opponent not in profiles

    for player in player_profiles:
        profile = player_profiles[player]
        profile["win_vs_lower"] = np.mean(profile["win_vs_lower"]) if len(profile["win_vs_lower"]) > 0 else 0
        profile["win_vs_higher"] = np.mean(profile["win_vs_higher"]) if len(profile["win_vs_higher"]) > 0 else 0
        profile["recent10"] = np.mean(profile["recent10"]) if len(profile["recent10"]) > 0 else 0
        profile['wvl_utr'] = np.mean(profile['wvl_utr']) if len(profile['wvl_utr']) > 0 else 0
        profile['wvh_utr'] = np.mean(profile['wvh_utr']) if len(profile['wvh_utr']) > 0 else 0
    return player_profiles

def get_player_profiles_general(data, history):
    player_profiles = {}

    for i in range(len(data)):
        for player, opponent in [(data['p1'][i], data['p2'][i]), (data['p2'][i], data['p1'][i])]:
            utr_diff = data['p1_utr'][i] - data['p2_utr'][i] if data['p1'][i] == player else data['p2_utr'][i] - data['p1_utr'][i]
            
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
                    "utr": history[opponent]['utr'],
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
                player_profiles[player]['h2h'][opponent] = [0,0,1,1]

            if player not in player_profiles[opponent]['h2h']:
                player_profiles[opponent]['h2h'][player] = [0,0,1,1]

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
                player_profiles[player]["win_vs_lower"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)
                player_profiles[player]["wvl_utr"].append(data["p2_utr"][i] if data["p1"][i] == player else data["p1_utr"][i])

            else:  # Player faced a higher-rated opponent
                player_profiles[player]["win_vs_higher"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)
                player_profiles[player]["wvh_utr"].append(data["p2_utr"][i] if data["p1"][i] == player else data["p1_utr"][i])

            if len(player_profiles[player]["recent10"]) < 10:
                player_profiles[player]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)
            else:
                player_profiles[player]["recent10"] = player_profiles[player]["recent10"][1:]
                player_profiles[player]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)

            if len(player_profiles[opponent]["recent10"]) < 10:
                player_profiles[opponent]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == opponent else data["p_win"][i] == 0)
            else:
                player_profiles[opponent]["recent10"] = player_profiles[opponent]["recent10"][1:]
                player_profiles[opponent]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == opponent else data["p_win"][i] == 0)
        # except:
        #     continue # player/opponent not in profiles

    for player in player_profiles:
        profile = player_profiles[player]
        profile["win_vs_lower"] = np.mean(profile["win_vs_lower"]) if len(profile["win_vs_lower"]) > 0 else 0
        profile["win_vs_higher"] = np.mean(profile["win_vs_higher"]) if len(profile["win_vs_higher"]) > 0 else 0
        profile["recent10"] = np.mean(profile["recent10"]) if len(profile["recent10"]) > 0 else 0
        profile['wvl_utr'] = np.mean(profile['wvl_utr']) if len(profile['wvl_utr']) > 0 else 0
        profile['wvh_utr'] = np.mean(profile['wvh_utr']) if len(profile['wvh_utr']) > 0 else 0
    return player_profiles

def get_player_history(utr_history):
    history = {}

    for i in range(len(utr_history)):
        if utr_history['first_name'][i]+' '+utr_history['last_name'][i][0] not in history:
            history[utr_history['first_name'][i]+' '+utr_history['last_name'][i][0]] = {
                'utr': utr_history['utr'][i]
            }

    return history

def make_prediction(player_1, player_2, location):
    # get data to fit to model    
    conn = st.connection('gcs', type=FilesConnection)
    data = conn.read("matches-scraper-bucket/atp_utr_tennis_matches.csv", input_format="csv", ttl=600)
    conn = st.connection('gcs', type=FilesConnection)
    utr_history = conn.read("utr_scraper_bucket/utr_history.csv", input_format="csv", ttl=600)

    x = np.empty(1)
    for i in range(len(data)):
        x = np.append(x, data['p1_utr'][i]-data['p2_utr'][i])

    x = x.reshape(-1,1)

    p = np.tanh(x) / 2 + 0.5
    model = LogitRegression()
    model.fit(0.9*x, p)

    p1 = player_1 # "Medvedev D."
    p2 = player_2 # "Alcaraz C."
    ps = [p1, p2]
    history = get_player_history(utr_history)
    player_profiles = get_player_profiles(data, history, ps[0], ps[1])

    score, p1_win, game_prop = get_score(ps, player_profiles, model)
    
    output_prediction = ""
    
    if p1_win:
        output_prediction = f'{p1} is predicted to win ({100*game_prop}% of games) against {p2}: '
    else:
        output_prediction = f'{p1} is predicted to lose ({100*(1-game_prop)}% of games) against {p2}:  '
    for i in range(len(score)):
        if i % 4 == 0 and int(score[i]) > int(score[i+2]):
            output_prediction += Fore.GREEN + score[i]
        elif i % 4 == 0 and int(score[i]) < int(score[i+2]):
            output_prediction += Fore.RED + score[i]
        else:
            output_prediction += score[i]
    
    return output_prediction
