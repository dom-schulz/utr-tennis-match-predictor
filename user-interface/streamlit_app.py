import streamlit as st
from openai import OpenAI
from pydantic import BaseModel
import json
import inspect
from st_files_connection import FilesConnection
import pandas as pd
from predict_utils import *
import matplotlib.pyplot as plt
from datetime import datetime
# from google.cloud import storage

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

# ========== Streamlit UI ==========
st.title("UTR Match Predictor1 🎾")

with st.sidebar:
    st.header("🔧 Tools & Insights")
    st.markdown("✅ Player Metrics")
    st.markdown("✅ Match Prediction")
    st.markdown("🚧 Tournament Tracker *(coming soon)*")
    st.markdown("🚧 Surface Win Rates *(coming soon)*")

# st.button("Create Custom Player Profile (Coming Soon)", disabled=True)

tabs = st.tabs(["🔮 Predictions", "📅 Upcoming Matches", "📈 Large UTR Moves", "🎾 Player Metrics", "ℹ️ About", "📣 Feedback"])

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

    # history = get_player_history(df)

    # content = []
    # for player in history.keys():
    #     row = {player: history[player]}

    # Optionally, sort or filter
    # df_sorted = df.sort_values(by="utr_change", ascending=False)
    # st.subheader("Top UTR Gains")
    # st.dataframe(df_sorted.head(10))

with tabs[3]:
    st.header("🎾 Player Metrics")

    # Load data from GCS
    conn = st.connection('gcs', type=FilesConnection)
    df1 = conn.read("utr_scraper_bucket/utr_history.csv", input_format="csv", ttl=600)
    df2 = conn.read("matches-scraper-bucket/atp_utr_tennis_matches.csv", input_format="csv", ttl=600)

    history = get_player_history(df1)
    player_df = get_player_profiles_general(df2, history)

    # Player selection
    st.subheader("Compare Two Players")
    player_names = [""] + sorted(player_df.keys())
    col1, col2 = st.columns(2)

    with col1:
        player1 = st.selectbox("Player 1", player_names, key="player1")

    with col2:
        player2 = st.selectbox("Player 2", player_names, key="player2")

    def display_player_metrics(player1, player2):
        if player1 != "" and player2 != "":
            profile = player_df[player1]
            utr_history = history.get(player1, {}).get("utr", [])
            dates = history.get(player1, {}).get("date", [])

            st.markdown(f"### {player1}")
            st.metric("Current UTR", round(profile.get("utr", 0), 2))
            st.metric("Win Rate Vs. Lower UTRs", f"{round(profile.get("win_vs_lower", 0) * 100, 2)}%")
            st.metric("Win Rate Vs. Higher UTRs", f"{round(profile.get("win_vs_higher", 0) * 100, 2)}%")
            st.metric("Win Rate Last 10 Matches", f"{round(profile.get("recent10", 0) * 100, 2)}%")
            try:
                st.metric("Head-To-Head (W-L)", f"{profile.get("h2h")[player2][0]} - {profile.get("h2h")[player2][1]-profile.get("h2h")[player2][0]}")
            except:
                st.metric("Head-To-Head (W-L)", "0 - 0")
            if utr_history and dates:
                chart_df = pd.DataFrame({
                    "Date": pd.to_datetime(dates),
                    "UTR": utr_history
                }).sort_values("Date")
                st.line_chart(chart_df.set_index("Date"))

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        display_player_metrics(player1, player2)
    with col2:
        display_player_metrics(player2, player1)

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
    We’re a team of student developers and tennis enthusiasts combining our passions for sports analytics, data science, and clean UI design. This is an ongoing project — we’re constantly improving predictions, cleaning data, and adding new insights.

    If you have feedback, want to contribute, or just love tennis tech, reach out!
    """)

with tabs[5]:
    st.header("💬 We Value Your Feedback!")
    ########## Feedback Function ###########
    def collect_feedback():
        pass
    # # Create a form to collect feedback
    # with st.form(key="feedback_form"):
    #     # Collect feedback from users
    #     rating = st.slider("How would you rate your experience?", min_value=1, max_value=5)
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