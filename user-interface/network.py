import numpy as np
import torch
import torch.nn as nn
from predict_utils import TennisPredictor

# Convert -1 values to 0 and create a mask
def create_masked_inputs(X):
    mask = (X != -1).float()
    X = torch.where(X == -1, torch.tensor(0.0), X)  # Replace -1 with 0
    return X, mask

# Preprocessing function to convert match data into features for the model
def preprocess_match_data(matches, profiles): # log_predict?
    match_vector = [matches['p1_utr']-matches['p2_utr'], 
                profiles[ matches['p1'] ]['win_vs_lower'],
                profiles[ matches['p2'] ]['win_vs_lower'],
                profiles[ matches['p1'] ]['win_vs_higher'],
                profiles[ matches['p2'] ]['win_vs_higher'],
                profiles[ matches['p1'] ]['recent10'],
                profiles[ matches['p2'] ]['recent10'],
                profiles[ matches['p1'] ]['wvl_utr'],
                profiles[ matches['p2'] ]['wvl_utr'],
                profiles[ matches['p1'] ]['wvh_utr'],
                profiles[ matches['p2'] ]['wvh_utr'],
                profiles[ matches['p1'] ]['h2h'][ matches['p2'] ][0] / profiles[ matches['p1'] ]['h2h'][ matches['p2'] ][1],
                profiles[ matches['p2'] ]['h2h'][ matches['p1'] ][0] / profiles[ matches['p2'] ]['h2h'][ matches['p1'] ][1],
                # profiles[ matches['p1'] ]['h2h'][ matches['p2'] ][2] / profiles[ matches['p1'] ]['h2h'][ matches['p2'] ][3],
                # profiles[ matches['p2'] ]['h2h'][ matches['p1'] ][2] / profiles[ matches['p2'] ]['h2h'][ matches['p1'] ][3]
                ]
    return match_vector

def get_player_profiles(data, history):
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
                    "utr": history[player], # Was history[player]['utr']
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
                    "utr": history[opponent], # Was history[opponent]['utr']
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
                # player_profiles[opponent]["win_vs_higher"].append(data["p_win"][i] == 1*data['p2_utr'][i] if data["p1"][i] == opponent else data["p_win"][i] == 0)

                player_profiles[player]["wvl_utr"].append(data["p2_utr"][i] if data["p1"][i] == player else data["p1_utr"][i])
                # player_profiles[opponent]["wvh_utr"].append(data["p2_utr"][i] == 1 if data["p1"][i] == opponent else data["p1_utr"][i])

            else:  # Player faced a higher-rated opponent
                player_profiles[player]["win_vs_higher"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)
                # player_profiles[opponent]["win_vs_lower"].append(data["p_win"][i] == 1*data['p2_utr'][i] if data["p1"][i] == opponent else data["p_win"][i] == 0)

                player_profiles[player]["wvh_utr"].append(data["p2_utr"][i] if data["p1"][i] == player else data["p1_utr"][i])
                # player_profiles[opponent]["wvl_utr"].append(data["p2_utr"][i] if data["p1"][i] == opponent else data["p1_utr"][i])

            if len(player_profiles[player]["recent10"]) < 10:
                player_profiles[player]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)
                # player_profiles[player]["r10_utr"].append(data["p2_utr"][i] if data["p1"][i] == player else data["p1_utr"][i])
            else:
                player_profiles[player]["recent10"] = player_profiles[player]["recent10"][1:]
                player_profiles[player]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)
                # player_profiles[player]["r10_utr"] = player_profiles[player]["r10_utr"][1:]
                # player_profiles[player]["r10_utr"].append(data["p2_utr"][i] if data["p1"][i] == player else data["p1_utr"][i])

            if len(player_profiles[opponent]["recent10"]) < 10:
                player_profiles[opponent]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == opponent else data["p_win"][i] == 0)
            else:
                player_profiles[opponent]["recent10"] = player_profiles[opponent]["recent10"][1:]
                player_profiles[opponent]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == opponent else data["p_win"][i] == 0)

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
        if utr_history['first_name'][i]+' '+utr_history['last_name'][i] not in history.keys():
            history[utr_history['first_name'][i]+' '+utr_history['last_name'][i]] = [[utr_history['utr'][i], utr_history['date'][i]]]
        else:
            history[utr_history['first_name'][i]+' '+utr_history['last_name'][i]].append([utr_history['utr'][i], utr_history['date'][i]])

    return history

def get_prop(model, matches, player_profiles):
    # Make one prediction
    X = preprocess_match_data(matches, player_profiles)
    # X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    y_test_one = model(X_tensor).squeeze().detach().numpy()

    return 1-float(y_test_one)