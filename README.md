# Perfect_Olympic_Predictors

Project Overview:

Our project aims to use Python to analyze and predict Paris 2024 Olympic medal winners!

Dependencies:
```py
pip install pandas plotly==5.22.0 openpyxl -U scikit-learn
```

We utilize naive bayes to predict the probabilty that a certain country will win a medal at the Paris 2024 Olympic games based on the datasets from the Tokyo 2020 Olympic games. We explore the influences of factors such as a country's probability of winning in the past (from which we only take into account the Tokyo games results due to limited datasets), and the percentage of teams sent by a country that won.

    A country's probability of winning in the past is calculated as the number of medals won by the country divided by the total number of medals possible to win (number of sporting disciplines multiplied by three (for gold, silver, and bronze medals)).

    The percentage of teams sent by a country that won is calculated by the number of medals won by the country divided by the number of teams the country sent to that Olympic game.

We used the following version of naive bayes theorem:

    P(A1 | B1, B2) = P(A1) * P(B1 | A1) * P(B2 | A1)
