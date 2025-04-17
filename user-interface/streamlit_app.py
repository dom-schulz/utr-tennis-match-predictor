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

        # get openai completion
        response = client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "user", "content": agent.instructions}] + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # handle tool calls

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools_map)

            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)

    # return new messages
    return messages[num_init_messages:]

# Execute tool function
def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    return tools_map[name](**args)


########### Tools ###############
# Tool function to check players
def gather_list_check_existence(player_1, player_2, location):
    """
    Reads UTR history, finds unique players, formats them as 'FirstName, LastName',
    and checks if the provided player_1 and player_2 exist in that list.
    Assumes player_1 and player_2 inputs are in 'FirstName, LastName' format.
    """
    player_list = []

    conn = st.connection('gcs', type=FilesConnection)
    try:
        # Read the file
        df_full = conn.read("utr_scraper_bucket/utr_history.csv", input_format="csv", ttl=600)

        # Ensure required columns exist
        if 'f_name' not in df_full.columns or 'l_name' not in df_full.columns:
             st.error("Required columns ('f_name', 'l_name') not found in utr_history.csv")
             # Return an error message that the agent might understand or pass back
             return "ERROR: Player data file is missing required columns."

        # Create DataFrame 'df' with unique names (handle potential missing values)
        df = df_full[['f_name', 'l_name']].dropna().drop_duplicates().reset_index(drop=True)

        # Append player list in "f_name l_name" format
        for row in df.itertuples(index=False):
            # Ensure names are strings before joining
            f_name_str = str(row.f_name)
            l_name_str = str(row.l_name)
            # print(f'{row.f_name} {row.l_name}')
            player_list.append(f"{f_name_str} {l_name_str}") # Combine names with a comma and space

    except Exception as e:
        st.error(f"Error reading or processing player data: {e}")
        # Return an error message
        return f"ERROR: Could not load or process player data. Details: {e}"

    # Check if the provided player names exist in the generated list
    p1_exists = player_1 in player_list
    p2_exists = player_2 in player_list

    if p1_exists and p2_exists:
        # Players found, return JSON
        return_json = json.dumps({"player_1": player_1, "player_2": player_2, "location": location})
        return return_json
    else:
        # One or both players not found, return invalid message
        missing = []
        if not p1_exists: missing.append(player_1)
        if not p2_exists: missing.append(player_2)
        # Provide feedback indicating the expected format might be the issue if names are missing
        return f"INVALID_PLAYERS: Could not find {', '.join(missing)}. Please ensure names are entered exactly as 'FirstName, LastName' (case-sensitive) and exist in the available data."




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
def set_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

tennis_court_bg_url = "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.istockphoto.com%2Fillustrations%2Ftennis-court&psig=AOvVaw0lVhJj0KLkLhRKgLmDqxld&ust=1744951115859000&source=images&cd=vfe&opi=89978449&ved=0CBkQjhxqFwoTCOiW6tKf3owDFQAAAAAdAAAAABAJ"
set_bg_image(tennis_court_bg_url)
st.title("🎾 Tennis Timmy Predictor 🤖") 
st.write("Enter two player names and a match location to receive a prediction for the match.")

st.divider()

# Ensure chat history persists across rerun
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
    st.divider()
# Second user input field at the bottom (keeping the same structure)
if user_query_2 := st.chat_input("Your request 2:"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query_2})
    with st.chat_message("user"):
        st.markdown(user_query_2)

    # Generate response for the second input (you might need different logic here)
    new_messages_2 = run_full_turn(get_agent, st.session_state.messages)

    # Append new messages to session history
    st.session_state.messages.extend(new_messages_2)

    # Display assistant response for the second input
    for msg in new_messages_2:
        role = msg.role if hasattr(msg, "role") else msg["role"]
        content = msg.content if hasattr(msg, "content") else msg["content"]

        if content is None or role == "tool" or role == "user":
            continue
        else:
            with st.chat_message(role):
                st.markdown(content)

