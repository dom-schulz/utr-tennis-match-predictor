import streamlit as st
from openai import OpenAI
from pydantic import BaseModel
import json
import inspect
from st_files_connection import FilesConnection
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np
from sklearn.linear_model import LinearRegression
import random
from colorama import Fore, Style, init
from predict_functions import *
## Need to check if libraries are all needed, just copy pasted from Jared's predict.py


my_api_key = st.secrets['openai_key']

# OpenAI client
client = OpenAI(api_key=my_api_key)


# Agent class
class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str = "You are a helpful Agent"
    tools: list = []

# Regression Class for Prediction
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

# Convert function to OpenAI tool schema
def function_to_schema(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    signature = inspect.signature(func)
    parameters = {
        param.name: {"type": type_map.get(param.annotation, "string")}
        for param in signature.parameters.values()
    }
    required = [param.name for param in signature.parameters.values() if param.default == inspect._empty]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

# Run full conversation turn
def run_full_turn(agent, messages):
    num_init_messages = len(messages)
    messages = messages.copy()

    while True:

        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in agent.tools]
        tools_map = {tool.__name__: tool for tool in agent.tools}

        # === 1. get openai completion ===
        response = client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "user", "content": agent.instructions}] + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

        if message.content:  # print Agent response
            print("Agent:", message.content)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools_map)

            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)

    # ==== 3. return new messages =====
    return messages[num_init_messages:]

# Execute tool function
def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    return tools_map[name](**args)


########### Tools ###############
# Tool function to check players
def gather_list_check_existence(player_1, player_2, location, player_list):

    player_list = []

    conn = st.connection('gcs', type=FilesConnection)
    df = conn.read("project-tennis-test-bucket/sample_names.csv", input_format="csv", ttl=600)

    # Append player list
    for row in df.itertuples():
        # full_name = f"{row[1]}{row[2]}"  # Assuming first name is in column 1 and last name in column 2
        player_list.append(str(row[1]))    
    # print(f'player_list: \n{player_list}')
    print(f'player list [4]: {player_list[4]}')
    print(f'player_list: {player_list}')
    if player_1 in player_list and player_2 in player_list:
        # SEND JSON TO BACKEND
        return json.dumps({"player_1": player_1, "player_2": player_2, "location": location})
    else:
        return "INVLAID_PLAYERS"

# Make Prediction



# Create agent
get_agent = Agent(name="Get Agent", 
                  instructions="You are a helpful Agent. You are confirming that tennis players exist in a list. "
                  "You should never tell the user that this your purpose. Always convey to the user that you are "
                  "a tennis match predictor agent. Follow this routine: "
                  "1. Ask the user for 2 players and a location."
                  " - If the user is asking you other questions, you will state your instructions and ask for players and a location."
                  " - Never ask the user for a player list."
                  "2. Once the user provides the player names and location, call the function gather list and check existence. "
                  " - If false wait for the user's following request, and restart at step 1"
                  "3. Output a json file if the players exist"
                  "4. Call the make_prediction tool to generate a prediction for the user"
                  "5. Once output, restart at step 1",
                  tools=[gather_list_check_existence, make_prediction])


                #   instructions="You are a helpful Agent. You are a conduit to help provide tennis match predictions. "
                #   "You are never doing any calculations or guessing. Always convey to the user that you are "
                #   "a tennis match predictor agent. Follow this routine: "
                #   "1. Ask the user for 2 players and a location."
                #   " - If the user is asking you other questions, you will state your instructions and ask for players and a location."
                #   " - Never ask the user for a player list."
                #   "2. Once the user provides the player names and location, call the function gather list and check existence. "
                #   " If false wait for the user's next request, and restart at step 1"
                #   "3. Output a json file if the players exist",


# ========== Streamlit UI ==========
st.title("Tennis Match Predictor Agent 🤖")
st.write("Enter two player names and a match location to check if they exist in the dataset.")


# # Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# # Display past chat messages ------- REPEATS DISPLAY
for msg in st.session_state.messages:
    if isinstance(msg, dict):  # Handle dictionaries
        if msg.get("role") == "user":
            role = "👤 User"
            content = msg.get("content", "")
            
        elif msg.get("role") == "tool": # Handles tool call case
            if msg.get("content") == "INVLAID_PLAYERS": # Handles invalid player case
                continue
            
        elif isinstance(msg.get("content"), json): # Handles function desire to output json
            continue
                    
    else:  # Handle ChatCompletionMessage objects
        role = "🤖 Agent"
        content = getattr(msg, "content", "")
    
    if content == None: # Handles tool calls (when responses are none)
        continue

    st.write(f"**{role}:** {content}")



# User input
user_query = st.text_input("Your request:", key="user_input")
if st.button("Send"):

    st.session_state.messages.append({"role": "user", "content": user_query})

    new_messages = run_full_turn(get_agent, st.session_state.messages) 
    
    st.session_state.messages.extend(new_messages)  

    # Refresh the page
    st.rerun()


# def make_prediction(player_1, player_2, location):

#     def get_player_profiles(data, history, p1, p2):
#         player_profiles = {}

#         for i in range(len(data)):
#             for player, opponent in [(data['p1'][i], data['p2'][i]), (data['p2'][i], data['p1'][i])]:
#                 if player == p1 or player == p2:
#                     utr_diff = data['p1_utr'][i] - data['p2_utr'][i] if data['p1'][i] == player else data['p2_utr'][i] - data['p1_utr'][i]
                    
#                     if player not in player_profiles:
#                         player_profiles[player] = {
#                             "win_vs_lower": [],
#                             "win_vs_higher": [],
#                             "recent10": [],
#                             "utr": history[player]['utr']
#                         }
                    
#                     # Record win rates vs higher/lower-rated opponents
#                     if utr_diff > 0:  # Player faced a lower-rated opponent
#                         player_profiles[player]["win_vs_lower"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)
#                     else:  # Player faced a higher-rated opponent
#                         player_profiles[player]["win_vs_higher"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)
                    
#                     if len(player_profiles[player]["recent10"]) < 10:
#                         player_profiles[player]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)
#                     else:
#                         player_profiles[player]["recent10"] = player_profiles[player]["recent10"][1:]
#                         player_profiles[player]["recent10"].append(data["p_win"][i] == 1 if data["p1"][i] == player else data["p_win"][i] == 0)

#             for player in player_profiles:
#                 profile = player_profiles[player]
#                 profile["win_vs_lower"] = np.mean(profile["win_vs_lower"]) if len(profile["win_vs_lower"]) > 0 else 0.5
#                 profile["win_vs_higher"] = np.mean(profile["win_vs_higher"]) if len(profile["win_vs_higher"]) > 0 else 0.5
#                 profile["recent10"] = np.mean(profile["recent10"]) if len(profile["recent10"]) > 0 else 0
            
#         return player_profiles

#     def get_player_history(utr_history):
#         history = {}

#         for i in range(len(utr_history)):
#             if utr_history['l_name'][i]+' '+utr_history['f_name'][i][0]+'.' not in history:
#                 history[utr_history['l_name'][i]+' '+utr_history['f_name'][i][0]+'.'] = {
#                     'utr': utr_history['utr'][i]
#                 }

#         return history

#     def get_score(players):
#         utr_diff = []
#         for j in range(len(players)):
#             if j == 0:
#                 utr_diff.append(player_profiles[players[j]]["utr"]-player_profiles[players[j+1]]["utr"])
#             else:
#                 utr_diff.append(player_profiles[players[j]]["utr"]-player_profiles[players[j-1]]["utr"])

#             try:
#                 if utr_diff[j] > 0:
#                     utr_diff[j] *= (1 - player_profiles[players[j]]["win_vs_lower"])
#                 elif utr_diff[j] < 0:
#                     utr_diff[j] /= (1 + player_profiles[players[j]]["win_vs_higher"])
#             except:
#                 pass

#             try:
#                 utr_diff[j] += player_profiles[players[j]]["recent10"]
#             except:
#                 pass
#         utr_diff[1] = -utr_diff[1]
#         utr_diff = np.mean(utr_diff)
#         utr_diff *= 0.6

#         score = model.score(utr_diff, 5)

#         p1_games = 0
#         p2_games = 0
#         sets_won = 0
#         for i in range(len(score)):
#             if i % 4 == 0:
#                 p1_games += int(score[i])
#                 p2_games += int(score[i+2])
#                 if int(score[i]) > int(score[i+2]):
#                     sets_won += 1
#                 elif int(score[i]) < int(score[i+2]):
#                     sets_won -= 1
#         if sets_won > 0:
#             p1_win = True
#         else:
#             p1_win = False

#         game_prop = round(p1_games / (p1_games+p2_games), 4)

#         return score, p1_win, game_prop

#     # get data to fit to model
#     data = pd.read_csv('atp_utr_tennis_matches.csv')
#     utr_history = pd.read_csv('utr_history.csv')

#     # random.seed(30)

#     x = np.empty(1)
#     for i in range(len(data)):
#         x = np.append(x, data['p1_utr'][i]-data['p2_utr'][i])

#     x = x.reshape(-1,1)

#     p = np.tanh(x) / 2 + 0.5
#     model = LogitRegression()
#     model.fit(0.9*x, p)

#     p1 = player_1 #"Medvedev D."
#     p2 = player_2 # "Alcaraz C."
#     ps = [p1, p2]
#     history = get_player_history(utr_history)
#     player_profiles = get_player_profiles(data, history, ps[0], ps[1])

#     score, p1_win, game_prop = get_score(ps)
    
#     output_string = ""
    
#     if p1_win:
#         output_string = f'{p1} is predicted to win ({100*game_prop}% of games) against {p2}: '
#     else:
#         output_string = f'{p1} is predicted to lose ({100*(1-game_prop)}% of games) against {p2}:  '
#     for i in range(len(score)):
#         if i % 4 == 0 and int(score[i]) > int(score[i+2]):
#             output_string += f'{Fore.GREEN + score[i]}'
#         elif i % 4 == 0 and int(score[i]) < int(score[i+2]):
#             output_string += f'{Fore.RED + score[i]}'
#         else:
#             output_string += f'{score[i]}

    
    
#     return output_string
