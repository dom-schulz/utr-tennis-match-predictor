# Tennis Match Predictor Project (Update: 4/6/25)

This repository contains two completed components for a tennis match prediction system:

## Current Items

### [Automated Scraper](https://github.com/dom-schulz/utr-tennis-match-predictor/tree/main/automated-utr-scraper)

The automated scraper collects historical match data from the Universal Tennis Rating (UTR) platform for professional tennis players. It's built on Google Cloud Platform with a fully automated pipeline utilizing:

- Cloud Scheduler for timing automation
- Cloud Pub/Sub for message passing
- Cloud Functions for triggering processes
- Compute Engine for running the scraping container (Docker)
- Cloud Storage for saving the scraped data

The system automatically runs twice weekly and intelligently shuts down after completion to minimize costs.

For more details, see the [Automated Scraper README](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/automated-utr-scraper/README.md). 


### [Streamlit App](https://github.com/dom-schulz/utr-tennis-match-predictor/tree/main/user-interface)

The Streamlit app serves as the user interface for the tennis match prediction system. It allows users to:

- Input player names to receive a prediction
- Receive generated match predictions based on UTR data via an AI chatbot
- View detailed match outcome probabilities and expected scores

The app is currently a functional prototype and will receive significant updates in mid-April. You can access the live interface at [Tennis Predictor App](https://utr-tennis-match-predictor.streamlit.app/).

For more details, see the [Streamlit App README](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/user-interface/README.md).


## Next Developments (April 2025)

### Automated Model Training

After the automated scraper runs, we plan to develop a training infrastructure to retrain the model automatically. This will allow the prediction system to:

- Use the latest scraped data without manual intervention
- Serve predictions by accessing the pre-trained model instead of retraining it for every request
- Upgrade from logistic regression to a neural network model, which has shown significant improvements in prediction accuracy during testing recently

### User Interface Updates

The Streamlit interface will receive significant improvements:

- Fixing display bugs with chat history
- Adding functionality for multiple simultaneous match predictions
- Simplifying the input process and improving player name format flexibility
- Expanding tennis information with additional tabs for match histories, upcoming matches, and player analytics
- Enhancing the visual design and user experience
- Optimizing performance through pre-trained model access 