# Tennis Match Predictor Project (5/7/25)

This repository contains our project to develop a tennis prediction system that collects player data through automated scrapers, analyzes match perforamnce, and delivers accurate match outcome predictions through an interactive web application. The system uses UTR (Universal Tennis Rating) data, historical match performances, and machine learning to provide probabilistic forecasts for professional tennis matches.


## User Interface

The interface for this project is deployed on a Streamlit app. It allows users to:

- Input player names to receive a match outcome prediction        
- View and compare some of the metrics influencing the prediction made
- View highest UTR changes in the past week

You can access the live interface at [Tennis Predictor App](https://utr-tennis-match-predictor.streamlit.app/)


## Predicition Details

One of the key metrics that sets our method apart from others is the use of UTR. But what exactly is it?

#### What is UTR?

The Universal Tennis Rating (UTR) is a standardized rating system designed to provide an objective measure of a tennis player's skill level. It is based on a player's performance in matches, taking into account the strength of their opponents and the outcomes of those matches. UTR is widely used in the tennis community to facilitate fair matchups, track player progress, and enhance competitive play.

**Key features of UTR include:**

- **Dynamic Rating**: A player's UTR can change frequently based on their match results, reflecting their current form and performance level.
- **Global Applicability**: UTR is used internationally, allowing players from different regions and levels to be compared fairly.
- **Comprehensive Data**: The rating system considers various factors, including match results, the quality of opponents, and the context of the matches played (e.g., tournament level).

Overall, UTR serves as a valuable tool for players, coaches, and tournament organizers, promoting a more structured and competitive environment in the sport of tennis. 

**For our altgorithm, it adds another complex metric that can provide another angle of insight into historical player performane**


### The Data 

The dataset utilizes many metrics focused on past player performance, including:

- UTR difference

- Player 1's win percentage versus lower UTRs across all historical matches

- Player 2's win percentage versus lower UTRs across all historical matches

- Player 1's win percentage versus higher UTRs across all historical matches

- Player 2's win percentage versus higher UTRs across all historical matches

- Player 1's win percentage in the most recent 10 matches

- Player 2's win percentage in the most recent 10 matches

- Player 1's average UTR beaten across all historical matches when UTR is lower (recorded as 0 if player 1 lost)

- Player 2's average UTR beaten across all historical matches when UTR is lower (recorded as 0 if player 2 lost)

- Player 1's average UTR beaten across all historical matches when UTR is higher (recorded as 0 if player 1 lost)

- Player 2's average UTR beaten across all historical matches when UTR is higher (recorded as 0 if player 2 lost)

- Player 1's win percentage against Player 2 historically

- Player 2's win percentage against Player 1 historically



The data utilized is scraped from the [UTR Website](https://www.utrsports.net/). This scraping process involves two programes that are automated on Google Cloud utilizing various services. One scraper pulls historical utr metrics and the other scrapes general match performance. For more details regrading the scraping implementation on Google Cloud, see the [automated-utr-scraper README](https://github.com/dom-schulz/utr-tennis-match-predictor/tree/main/automated-utr-scraper) 


### Algorithm Details

```python
class TennisPredictor(nn.Module):
    def __init__(self, input_size):
        super(TennisPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 1028)
        self.fc2 = nn.Linear(1028, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.sigmoid(self.fc8(x))
        return x
```

#### Summary of Activation Functions Used in `TennisPredictor`

1. **ReLU (Rectified Linear Unit)**:
   - **Usage**: Applied in hidden layers (`fc1` to `fc7`).
   - **Reasons for Choice**:
     - **Non-linearity**: Introduces non-linearity, enabling the model to learn complex patterns.
     - **Sparsity**: Outputs zero for negative inputs, promoting sparse representations and reducing overfitting.
     - **Efficiency**: Computationally efficient due to simple thresholding at zero.

2. **Sigmoid**:
   - **Usage**: Used in the output layer (`fc8`).
   - **Reasons for Choice**:
     - **Output Range**: Produces values between 0 and 1, suitable for binary classification (ie. win or lose).
     - **Interpretability**: Outputs can be interpreted as probabilities, ideal for predicting match outcomes.

The combination of `ReLU` in hidden layers and `Sigmoid` in the output layer allows the `TennisPredictor` model to effectively learn complex relationships while providing clear probabilistic predictions for match outcomes.

#### Automated Model Training

After each iteration of the automated scrapers, the model is retrained automatically. This allows the prediction system to:

- Use the latest scraped data without manual intervention
- Serve predictions by accessing the pre-trained model instead of retraining it for every request
- Upgrade from logistic regression to a neural network model, which has shown significant improvements in prediction accuracy during recent testing
