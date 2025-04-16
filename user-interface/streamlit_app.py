import streamlit as st
from openai import OpenAI
from pydantic import BaseModel
import json
import inspect
from st_files_connection import FilesConnection
import pandas as pd
from predict_utils import *

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
    df = conn.read("project-tennis-test-bucket/sample_names.csv", input_format="csv", ttl=600)

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
st.title("UTR Match Predictor Test 🤖")

tabs = st.tabs(["🔮 Predictions", "📅 Upcoming Matches", "📈 Large UTR Moves", "ℹ️ About"])

with tabs[0]:
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
            st.markdown(f"""
        <div style="background-color:#f8f9fa; border-radius:12px; padding:20px; box-shadow:0 2px 10px rgba(0,0,0,0.05);">
            <h4 style="color:#003366;">🎾 Prediction</h4>
            <pre style="white-space: pre-wrap; font-family: 'Courier New', monospace;">{content}</pre>
        </div>
        """,
        unsafe_allow_html=True)
    
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
    st.write("Here you can display upcoming tennis matches (e.g., from a dataset or API).")

# === Tab: Large UTR Moves ===
with tabs[2]:
    st.header("📈 Large UTR Moves")
    st.write("This tab will highlight matches where players gained or lost a large amount of UTR.")

    # Load the CSV from your bucket
    # conn = st.connection('gcs', type=FilesConnection)
    # df = conn.read("project-tennis-test-bucket/utr_history.csv", input_format="csv", ttl=600)

    # # Show the top few rows
    # st.dataframe(df)

    # history = get_player_history(df)

    # content = []
    # for i in range(len(df)):
        

    # Optionally, sort or filter
    # df_sorted = df.sort_values(by="utr_change", ascending=False)
    # st.subheader("Top UTR Gains")
    # st.dataframe(df_sorted.head(10))
