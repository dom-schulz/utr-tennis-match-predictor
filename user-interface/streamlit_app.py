import streamlit as st
from openai import OpenAI
from pydantic import BaseModel
import json
import inspect
from st_files_connection import FilesConnection
import pandas as pd
from predict_utils import *
import matplotlib.pyplot as plt

# OpenAI client
my_api_key = st.secrets['openai_key']
client = OpenAI(api_key=my_api_key)


# Agent class
class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str = "You are a helpful Agent"
    tools: list = []


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

        # 1. get openai completion
        response = client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "user", "content": agent.instructions}] + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # 2 handle tool calls

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools_map)

            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)

    # 3. return new messages
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
    df = conn.read("utr_scraper_bucket/sample_names.csv", input_format="csv", ttl=600)

    # Append player list
    for row in df.itertuples():
        player_list.append(str(row[1]))    

    if player_1 in player_list and player_2 in player_list:
        # SEND JSON TO BACKEND
        return_json = json.dumps({"player_1": player_1, "player_2": player_2, "location": location})
        return return_json
    else:
        return "INVALID_PLAYERS"


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
                "4. Call the make_prediction tool to generate a prediction for the user. Make an ouput with the following format:"
                "Prediction: \n"
                        "Jacquet K. is predicted to lose (49.09% of games) against Collignon R. \n\n"
                        "Predicted scorelines: \n"
                            "6-1 \n"
                            "7-6\n"
                            "4-6\n"
                            "5-7\n"
                            "6-7\n"
                        "If you have another match in mind, please provide the names of two players and the location!"
                  "5. Once output, restart at step 1",
                  tools=[gather_list_check_existence, make_prediction])

# ======== Streamlit Theme ======== #
# st.markdown("""
#     <style>
#     /* Custom color theme */
#     .main {
#         background-color: #f7f9fc;
#     }
#     h1, h2, h3 {
#         color: #1a2b4c;
#     }
#     .block-container {
#         padding-top: 2rem;
#     }
#     /* Cards or sections */
#     .stTabs [data-baseweb="tab"] {
#         background-color: #e6eef9;
#         color: #0f1e3d;
#         font-weight: 600;
#     }
#     .stTabs [aria-selected="true"] {
#         background-color: #1a2b4c;
#         color: white;
#     }
#     </style>
# """, unsafe_allow_html=True)

# ========== Streamlit UI ==========
st.title("UTR Match Predictor Test 🤖")

with st.sidebar:
    st.header("🔧 Tools & Insights")
    st.markdown("✅ Player Comparison")
    st.markdown("✅ Match Prediction")
    st.markdown("🚧 Tournament Tracker *(coming soon)*")
    st.markdown("🚧 Surface Win Rates *(coming soon)*")

# st.button("Create Custom Player Profile (Coming Soon)", disabled=True)

tabs = st.tabs(["🔮 Predictions", "📅 Upcoming Matches", "📈 Large UTR Moves", "UTR Graph", "ℹ️ About"])

with tabs[0]:
    st.subheader("AI-Powered Match Outcome Predictor")
    st.caption("Leverage player data and win percentages to simulate match outcomes in seconds.")

    st.write("Enter two player names and a match location to receive a prediction for the match.")
    
    # Ensure chat history persists across reruns
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history using `st.chat_message`
    for msg in st.session_state.messages:
        if isinstance(msg, dict):  # If dict message type
            if msg['role'] == "user":  # User messages produce message to output
                content = msg.get("content")
                role = "user"
            elif msg['role'] == "tool":  # Tool calls don't produce message to output
                continue
            else:  # Error, produce role
                raise ValueError(f"Invalid dictionary role: {msg['role']}")
        else:  # Handles ChatCompletionMessage object
            if msg.role == "assistant":
                content = msg.content
                role = "assistant"
                if content is None:  # Skip displaying None content
                    continue
            else:  # Error, produce role
                raise ValueError(f"Invalid ChatCompletionMessage role: {msg['role']}")
    
        # Display message
        with st.chat_message(role):
            st.markdown(content)
    
    # User input field at the bottom
    if user_query := st.chat_input("Your request:"):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
    
        # Generate response
        new_messages = run_full_turn(get_agent, st.session_state.messages)
    
        # Append new messages to session history without altering prior assistant messages
        st.session_state.messages.extend(new_messages)
    
        # Display assistant response
        for msg in new_messages:
            role = msg.role if hasattr(msg, "role") else msg["role"]
            content = msg.content if hasattr(msg, "content") else msg["content"]
    
            if content is None or role == "tool" or role == "user":
                continue  # Skip None content, tool responses, or user input
            else:
                with st.chat_message(role):
                    st.markdown(content)
                    
# === Tab: Upcoming Matches ===
with tabs[1]:
    st.header("📅 Upcoming Matches")
    st.subheader("Stay Ahead of the Game")
    st.caption("See what's next on the pro circuit, and who’s most likely to rise.")

    st.write("Here you can display upcoming tennis matches (e.g., from a dataset or API).")

# === Tab: Large UTR Moves ===
with tabs[2]:
    st.header("📈 Large UTR Moves")
    st.subheader("Biggest Shifts in Player Ratings")
    st.caption("Our algorithm tracks the highest-impact UTR swings — who’s peaking, who’s slipping.")

    st.write("This tab will highlight matches where players gained or lost a large amount of UTR since the previous week.")

    # Load the CSV from your bucket
    conn = st.connection('gcs', type=FilesConnection)
    df = conn.read("utr_scraper_bucket/utr_history.csv", input_format="csv", ttl=600)

    # # Show the top few rows
    # st.dataframe(df.head(10))

    # content = []
    # prev_name = ''
    # for i in range(len(df)):
    #     if df['utr'][i] > 13:
    #         curr_name = df['f_name'][i]+' '+df['l_name'][i]
    #         if curr_name != prev_name:
    #             curr_name = df['f_name'][i]+' '+df['l_name'][i]
    #             content.append([df['f_name'][i]+' '+df['l_name'][i], df['utr'][i+1], df['utr'][i], 
    #                             df['utr'][i]-df['utr'][i+1], 100*((df['utr'][i]/df['utr'][i+1])-1)])
    #         prev_name = curr_name
    # df = pd.DataFrame(content, columns=["Name", "Previous UTR", "Current UTR", "UTR Change", "UTR % Change"])
    # df = df.sort_values(by="UTR % Change", ascending=False)
    # st.dataframe(df.head(10))

    '''# df = df.sort_values(by="UTR % Change", ascending=True)
    # st.dataframe(df.head(10))'''

    # history = get_player_history(df)

    # content = []
    # for player in history.keys():
    #     row = {player: history[player]}

    # Optionally, sort or filter
    # df_sorted = df.sort_values(by="utr_change", ascending=False)
    # st.subheader("Top UTR Gains")
    # st.dataframe(df_sorted.head(10))

with tabs[3]:
    # Load data from GCS
    conn = st.connection('gcs', type=FilesConnection)
    df = conn.read("matches-scraper-bucket/atp_utr_tennis_matches.csv", input_format="csv", ttl=600)
    df = df[-100:]

    # Example scatter plot
    fig, ax = plt.subplots()
    colors = df['p_win'].map({1: 'blue', 0: 'red'})  # Adjust depending on how p_win is encoded
    ax.scatter(df['p1_utr'], df['p2_utr'], c=colors)
    ax.set_xlabel("Player 1 UTR")
    ax.set_ylabel("Player 2 UTR")
    ax.set_title("UTR Matchups by Outcome (R=p1w, B=p2w)")

    st.pyplot(fig)

with tabs[4]:
    st.markdown("""
    ### What is MatchMind?
    **MatchMind** is your smart tennis insights assistant.  
    Powered by OpenAI and match data, it gives fast, intuitive predictions — so you can always know who’s got the edge.

    **Built for players, fans, and data nerds alike.**
    """)