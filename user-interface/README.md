# Tennis Match Predictor - Streamlit Interface (3/6/25)

## Interface Overview

This interface utilizes [Streamlit](https://streamlit.io/) to create an interactive web application that predicts tennis match outcomes using UTR (Universal Tennis Rating) data and an AI agent. The current interface is a functional skeleton, with the last substantial update on March 6, 2024, and serves as a proof-of-concept for our tennis prediction system. We plan to update and improve this interface in mid April.

The app connects to Google Cloud Storage to access match history data and player information. Rather than using pre-trained models, the application trains a custom logistic regression model on-demand for each prediction (this model includes randomness which can slightly alter the model's predictions of the same match).

### Try it yourself

You can access the app at: [Tennis Predictor App](https://utr-tennis-match-predictor.streamlit.app/)

**Note**: If the app becomes unresponsive, look for the "Rerun" button in the upper right corner of the interface to restart it.

## Core Files

[`streamlit_app.py`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/user-interface/streamlit_app.py)

This is the main application file that creates the web interface and handles user interactions. It uses Streamlit's components to render the UI and manages the conversation flow with the OpenAI-powered chat assistant.

[`predict_utils.py`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/user-interface/predict_utils.py)

This utility file contains the machine learning model and prediction logic. About 80% of this file was created by a classmate, but I heavily modified the `make_prediction` function to integrate with Google Cloud Storage for data access and to format predictions in a way that works with the OpenAI API. We are currently working with different models with better performance and will integrate in mid to late April as well. 

## Secret Management

The application uses Streamlit's built-in secrets management system to handle sensitive information:

```python
# Accessing the OpenAI API key from secrets
my_api_key = st.secrets['openai_key']
client = OpenAI(api_key=my_api_key)
```

Key secrets managed through this system include:
- Google Cloud Storage credentials for accessing tennis match data
- Any other service credentials required for the application


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