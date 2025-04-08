# Tennis Match Predictor - Streamlit Interface (3/6/25)

## Interface Overview

This interface utilizes [Streamlit](https://streamlit.io/) to create an interactive web application that predicts tennis match outcomes using UTR (Universal Tennis Rating) data and an AI agent. The current interface is a functional skeleton, with the last substantial update on March 6, 2024, and serves as a proof-of-concept for our tennis prediction system. We plan to update and improve this interface in mid April.

The app connects to Google Cloud Storage to access match history data and player information. Rather than using pre-trained models, the application trains a custom logistic regression model on-demand for each prediction (this model includes randomness which can slightly alter the model's predictions of the same match).

### Try it yourself

You can access the app at: [Tennis Predictor App](https://utr-tennis-match-predictor.streamlit.app/)

If you want to test the app with sample inputs, try these player names and location:
```
Medvedev D. Alcaraz C. Seattle
```

**Note**: If the app becomes unresponsive, look for the "Rerun" button in the upper right corner of the interface to restart it.

## Core Files

[`streamlit_app.py`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/user-interface/streamlit_app.py)

This is the main application file that creates the web interface and handles user interactions. It uses Streamlit's components to render the UI and manages the conversation flow with the OpenAI-powered chat assistant.

Key components:
- Streamlit UI elements (title, chat interface)
- OpenAI API integration
- Agent configuration and tool management
- Session state handling for chat history

```python
# Agent class using Pydantic for structure
class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str = "You are a helpful Agent"
    tools: list = []

# Create the tennis prediction agent
get_agent = Agent(name="Get Agent", 
                instructions="You are a helpful Agent. You are confirming that tennis players exist in a list. "
                "You should never tell the user that this your purpose. Always convey to the user that you are "
                "a tennis match predictor agent. Follow this routine: "
                # ... instructions continue ...
                tools=[gather_list_check_existence, make_prediction])
```

[`predict_utils.py`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/user-interface/predict_utils.py)

This utility file contains the machine learning model and prediction logic. About 80% of this file was created by a classmate, but I heavily modified the `make_prediction` function to integrate with Google Cloud Storage for data access and to format predictions in a way that works with the OpenAI API. We are currently working with different models with better performance and will integrate in mid to late April as well. 

Key components:
- Custom `LogitRegression` class for probability predictions
- Player profile generation and analysis
- Match score simulation
- Google Cloud Storage integration for data access

```python
def make_prediction(player_1, player_2, location):
    # get data to fit to model    
    conn = st.connection('gcs', type=FilesConnection)
    data = conn.read("project-tennis-test-bucket/atp_utr_tennis_matches.csv", input_format="csv", ttl=600)
    conn = st.connection('gcs', type=FilesConnection)
    utr_history = conn.read("project-tennis-test-bucket/utr_history.csv", input_format="csv", ttl=600)
    
    # Model training and prediction logic
    # ...
    
    output_prediction = f'{p1} is predicted to {"win" if p1_win else "lose"} ({100*game_prop}% of games) against {p2}: '
    # ... format score details ...
    
    return output_prediction
```

`secrets.toml` (not in repository)

This file contains sensitive information such as API keys and cloud service credentials. It is automatically excluded from version control via `.gitignore` for security reasons. This file's contents are input into Streamlit's built in secret manager.

## OpenAI API Implementation

The application uses OpenAI's API to create an interactive conversation powered by an AI agent. The agent is configured with specific instructions and tools that allow it to perform specialized tasks.

### Agent Architecture

The core of our implementation uses a custom Agent class that encapsulates:
1. The specific GPT model to use (gpt-4o-mini) 
2. Detailed instructions on how to interact with users (these instructions can be edited by agent)
3. A set of specialized tools the agent can use

```python
get_agent = Agent(
    name="Get Agent", 
    instructions="You are a helpful Agent...",
    tools=[gather_list_check_existence, make_prediction]
)
```

### Function Calling

One of the key features is the use of OpenAI's function calling capability. Python functions are converted into a format that the OpenAI API can understand and execute:

```python
def function_to_schema(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        # ... other type mappings ...
    }
    
    signature = inspect.signature(func)
    parameters = {
        param.name: {"type": type_map.get(param.annotation, "string")}
        for param in signature.parameters.values()
    }
    # ... additional processing ...
    
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
```

This allows the agent to:
1. Validate player names against our database
2. Generate tennis match predictions based on UTR ratings and historical data
3. Present the results in a user-friendly format

## Secret Management

The application uses Streamlit's built-in secrets management system to handle sensitive information:

```python
# Accessing the OpenAI API key from secrets
my_api_key = st.secrets['openai_key']
client = OpenAI(api_key=my_api_key)
```

Key secrets managed through this system include:
- OpenAI API key for accessing GPT models
- Google Cloud Storage credentials for accessing tennis match data
- Any other service credentials required for the application


## Challenges

### OpenAI API Integration

Working with the OpenAI API presented several challenges:

1. **Instruction Design**: Crafting instructions for the agent that covers all edge cases without being overly verbose.

2. **Hallucination**: Understanding where the model might "hallucinate" information and implementing safeguards. The agent can read and interpret code correctly, but sometimes extrapolates or invents content when unsure.

3. **Response Formatting**: Ensuring that the model consistently returns predictions in the expected format for display to users.

4. **Tool Integration**: Combining the function-calling capabilities with custom Python functions in a way that produced reliable results.

5. **Error Handling**: Developing error handling for cases where the API might not respond as expected or where user inputs might be invalid.

## Future Plans/Improvements

The current implementation serves as a starting point, with several planned improvements:

1. **Fix Display Bug**: Address an issue where previous predictions are displayed again in the chat history until a new prediction is made.

2. **Multiple Predictions**: Enable the ability to compare multiple match predictions simultaneously.

3. **Simplify Inputs**: Remove the location field since it's currently not being used in the prediction algorithm.

4. **Name Format Flexibility**: Improve the player name input system to be more flexible with formats, rather than requiring last name and first initial.

5. **Expanded Tennis Information**: Add additional tabs to display various tennis-related information:
   - Past match histories
   - Upcoming matches and tournaments
   - Recent upsets in the tennis world
   - Current UTR rankings
   - Player performance analytics

6. **UI/UX Improvements**: Enhance the visual design and user experience of the application to make it more intuitive and engaging.

7. **Performance Optimization**: Pre-train models and store them rather than training on each prediction request to improve response times. 