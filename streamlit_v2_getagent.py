import streamlit as st
from openai import OpenAI
from pydantic import BaseModel
import json
import inspect

# OpenAI client
client = OpenAI(api_key="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")


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


# Tool function to check players
def gather_list_check_existence(player_1, player_2, location):
    player_list = [
        "Ryan Smith", "John Doe", "Roger Federer", "Rafael Nadal",
        "Novak Djokovic", "Serena Williams", "Naomi Osaka", "Ashleigh Barty"
    ]

    if player_1 in player_list and player_2 in player_list:
        # SEND JSON TO BACKEND
        return json.dumps({"player_1": player_1, "player_2": player_2, "location": location})
    else:
        return "INVLAID_PLAYERS"


# Create agent
get_agent = Agent(name="Get Agent", 
                  instructions="You are a helpful Agent. You are confirming that tennis players exist in a list. "
                  "You should never tell the user that this your purpose. Always convey to the user that you are "
                  "a tennis match predictor agent. Follow this routine: "
                  "1. Ask the user for 2 players and a location."
                  " - If the user is asking you other questions, you will state your instructions and ask for players and a location."
                  " - Never ask the user for a player list."
                  "2. Once the user provides the player names and location, call the function gather list and check existence. "
                  " If false wait for the user's next request, and restart at step 1"
                  "3. Output a json file if the players exist",
                  tools=[gather_list_check_existence])


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
    # print(f'content query append: {user_query}\n')
    st.session_state.messages.append({"role": "user", "content": user_query})

    new_messages = run_full_turn(get_agent, st.session_state.messages) 

    # print(F'new messages: {new_messages[1:]}\n\n')
    
    st.session_state.messages.extend(new_messages)  

    # print(f'st session state messages: {st.session_state.messages}\n\n')

    # Refresh the page
    st.rerun()
