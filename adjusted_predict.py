import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np
from sklearn.linear_model import LinearRegression
import random
from colorama import Fore, Style, init
from st_files_connection import FilesConnection
import streamlit as st


'''
MODELS TESTED:
 - Pseudo Logistic Regression Using tanh()  | Current Version
 - Logistic Regression via Scikit Learn     | Unable to assign proportion (only 0 or 1)
 - Random Forests via Scikit Learn          | Unable to assign proportion (only 0 or 1)
'''

class LogitRegression(LinearRegression):

    def fit(self, x, p):
        p = np.asarray(p)
        y = np.log(p / (1 - p))
        return super().fit(x, y)

    def predict(self, x):
        y = super().predict(x)
        return 1 / (np.exp(-y) + 1)
    
    def score(self, utr_diff, best_of):
        prop = self.predict([[utr_diff]])[0][0]
        score = ''
        sets_won = 0
        num_sets = 0
        for _ in range(best_of):
            p1_games = 0
            p2_games = 0
            done = True
            while done:
                if p1_games == 6 and p2_games < 5 or p2_games == 6 and p1_games < 5:
                    break
                elif p1_games == 7 or p2_games == 7:
                    break
                val = random.uniform(0,1)
                if val < prop:
                    p1_games += 1
                else:
                    p2_games += 1

            num_sets += 1
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
        return score

    def profile(self, data):
        profile = []

        for i in range(len(data)):
            pass

        return profile

def get_player_profiles(data, history, p1, p2):
    player_profiles = {}

    for i in range(len(data)):
        for player, opponent in [(data['p1'][i], data['p2'][i]), (data['p2'][i], data['p1'][i])]:
            if player == p1 or player == p2:
                utr_diff = data['p1_utr'][i] - data['p2_utr'][i] if data['p1'][i] == player else data['p2_utr'][i] - data['p1_utr'][i]
                print(f'history: {history}')
                if player not in player_profiles:
                    player_profiles[player] = {
                        "win_vs_lower": [],
                        "win_vs_higher": [],
                        "recent10": [],
                        "utr": history[player]['utr']
                    }
                
                # Record win rates vs higher/lower-rated opponents
                if utr_diff > 0:  # Player faced a lower-rated opponent
                    player_profiles[player]["win_vs_lower"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)
                else:  # Player faced a higher-rated opponent
                    player_profiles[player]["win_vs_higher"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)
                
                if len(player_profiles[player]["recent10"]) < 10:
                    player_profiles[player]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)
                else:
                    player_profiles[player]["recent10"] = player_profiles[player]["recent10"][1:]
                    player_profiles[player]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)

    for player in player_profiles:
        profile = player_profiles[player]
        profile["win_vs_lower"] = np.mean(profile["win_vs_lower"]) if len(profile["win_vs_lower"]) > 0 else 0.5
        profile["win_vs_higher"] = np.mean(profile["win_vs_higher"]) if len(profile["win_vs_higher"]) > 0 else 0.5
        profile["recent10"] = np.mean(profile["recent10"]) if len(profile["recent10"]) > 0 else 0
    
    return player_profiles

def get_player_history(utr_history):
    history = {}

    for i in range(len(utr_history)):
        if utr_history['l_name'][i]+' '+utr_history['f_name'][i][0]+'.' not in history:
            history[utr_history['l_name'][i]+' '+utr_history['f_name'][i][0]+'.'] = {
                'utr': utr_history['utr'][i]
            }

    return history

def get_score(players, player_profiles, model):
    utr_diff = []
    for j in range(len(players)):
        if j == 0:
            utr_diff.append(player_profiles[players[j]]["utr"]-player_profiles[players[j+1]]["utr"])
        else:
            utr_diff.append(player_profiles[players[j]]["utr"]-player_profiles[players[j-1]]["utr"])

        try:
            if utr_diff[j] > 0:
                utr_diff[j] *= (1 - player_profiles[players[j]]["win_vs_lower"])
            elif utr_diff[j] < 0:
                utr_diff[j] /= (1 + player_profiles[players[j]]["win_vs_higher"])
        except:
            pass

        try:
            utr_diff[j] += player_profiles[players[j]]["recent10"]
        except:
            pass
    utr_diff[1] = -utr_diff[1]
    utr_diff = np.mean(utr_diff)
    utr_diff *= 0.6

    score = model.score(utr_diff, 5)

    p1_games = 0
    p2_games = 0
    sets_won = 0
    for i in range(len(score)):
        if i % 4 == 0:
            p1_games += int(score[i])
            p2_games += int(score[i+2])
            if int(score[i]) > int(score[i+2]):
                sets_won += 1
            elif int(score[i]) < int(score[i+2]):
                sets_won -= 1
    if sets_won > 0:
        p1_win = True
    else:
        p1_win = False

    game_prop = round(p1_games / (p1_games+p2_games), 4)

    return score, p1_win, game_prop


def make_prediction(player_1, player_2, location):
    # get data to fit to model    
    conn = st.connection('gcs', type=FilesConnection)
    data = conn.read("project-tennis-test-bucket/atp_utr_tennis_matches.csv", input_format="csv", ttl=600)
    conn = st.connection('gcs', type=FilesConnection)
    utr_history = conn.read("project-tennis-test-bucket/utr_history.csv", input_format="csv", ttl=600)

    # random.seed(30)

    x = np.empty(1)
    for i in range(len(data)):
        x = np.append(x, data['p1_utr'][i]-data['p2_utr'][i])

    x = x.reshape(-1,1)

    p = np.tanh(x) / 2 + 0.5
    model = LogitRegression()
    model.fit(0.9*x, p)

    p1 = player_1 #"Medvedev D."
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
    # print()
    # init(autoreset=True)
    
    return output_prediction

# '''
# EXAMPLE OUTPUT:

# +-----------------------------------------------------------------------------------------+
# |                                                                                         |
# |   Medvedev D. is predicted to lose (63.33% of games) against Alcaraz C.:  1-6 6-7 4-6   |
# |                                                                                         |
# +-----------------------------------------------------------------------------------------+

# Output will always be player 1's score first. In this case I had 
# player 1 as Medvedev and player 2 as Alcaraz. So in this example 
# Medvedev lost to Alcaraz.