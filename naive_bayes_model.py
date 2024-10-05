import pandas as pd
import numpy as np
import plotly.express as px
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


def main():
    athletes_file = "./olympic2021data/Athletes.xlsx"
    medals_file = "./olympic2021data/Medals.xlsx"

    # Cleaning teams dataset
    df_athletes = pd.read_excel(athletes_file, sheet_name="Details")
    df_athletes = df_athletes.drop(['Name'], axis=1)

    # Cleaning medals dataset
    df_medals = pd.read_excel(medals_file, sheet_name="Details")
    df_medals = df_medals.drop(['Rank', 'Rank by Total', 'Gold', 'Silver', "Bronze"], axis=1)

    # Calculating probs
    # ## Probability that a team wins a medal
    # country_athletes = df_athletes.groupby('NOC').count().rename(columns={'Discipline': 'Teams'})
    country_athletes = defaultdict(lambda: set())
    for row in df_athletes.itertuples():
        country = row.NOC
        discipline = row.Discipline
        country_athletes[country].add(discipline)

    total_medals = df_medals['Total'].sum()

    # How to get info from each dataset
    # print(country_teams.loc['Argentina', 'Teams'])
    # print(df_medals.loc[df_medals['Team/NOC'] == 'Japan', 'Total'].values[0])

    final_data = {}
    y_axis = []
    x_axis = []
    for key in country_athletes:
        country = key
        num_events_participated_in = len(country_athletes[key])
        num_medals = df_medals.loc[df_medals['Team/NOC'] == country, 'Total'].values
        if (len(num_medals) > 0):
            num_medals = num_medals[0]
        else:
            num_medals = 0
        # print("Country: " + country + "; Number of events: " + str(num_events_participated_in) + "; Medals: " + str(num_medals))

        # Naive Bayes calculation variables
        p_a1 = 1 if num_medals > 0 else 0
        p_b1_a1 = num_medals/(num_events_participated_in * 3)
        p_b2_a1 = num_medals/num_events_participated_in if num_medals/num_events_participated_in <= 1 else 1

        # Naive Bayes learning model dataset
        x_axis_dataset = []
        if (p_a1 > 0):
            y_axis.append(1)
            x_axis_dataset.append(1)
        else:
            y_axis.append(0)
            x_axis_dataset.append(0)

        if (p_b1_a1 < 0.2):
            x_axis_dataset.append(0)
        elif (p_b1_a1 < 0.4):
            x_axis_dataset.append(1)
        elif (p_b1_a1 < 0.6):
            x_axis_dataset.append(2)
        elif (p_b1_a1 < 0.8):
            x_axis_dataset.append(3)
        else:
            x_axis_dataset.append(4)

        if (p_b2_a1 < 0.2):
            x_axis_dataset.append(0)
        elif (p_b2_a1 < 0.4):
            x_axis_dataset.append(1)
        elif (p_b2_a1 < 0.6):
            x_axis_dataset.append(2)
        elif (p_b2_a1 < 0.8):
            x_axis_dataset.append(3)
        else:
            x_axis_dataset.append(4)

        x_axis.append(x_axis_dataset)

        # Naive Bayes Applied thereom
        final_data[country] = p_a1 * p_b1_a1 * p_b2_a1

    # Naive Bayes learning models
    x_train, x_test, y_train, y_test = train_test_split(x_axis, y_axis, test_size=0.4, random_state=42)
    # MULTINOMIAL NB
    multinomial_model = MultinomialNB()
    multinomial_model.fit(x_test, y_test)
    multinomial_y_predictions = multinomial_model.predict(x_test)
    multinomial_accuracy = accuracy_score(y_test, multinomial_y_predictions)
    print("MULTINOMIAL NAIVE-BAYES MODEL")
    print("Overall Accuracy: ", multinomial_accuracy)
    print("Overall classification report:")
    print(classification_report(
        y_test, multinomial_y_predictions, zero_division=0))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, multinomial_y_predictions))

    # GAUSSIAN NB
    gaussian_model = GaussianNB()
    gaussian_model.fit(x_train, y_train)
    gaussian_y_predictions = gaussian_model.predict(x_test)
    gaussian_accuracy = accuracy_score(y_test, gaussian_y_predictions)
    print("\n\nGAUSSIAN NAIVE-BAYES MODEL")
    print("Overall Accuracy: ", gaussian_accuracy)
    print("Overall classification report:")
    print(classification_report(
        y_test, gaussian_y_predictions, zero_division=0))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, gaussian_y_predictions))

    # GAUSSIAN NB
    bernoulli_model = BernoulliNB()
    bernoulli_model.fit(x_train, y_train)
    bernoulli_y_predictions = bernoulli_model.predict(x_test)
    bernoulli_accuracy = accuracy_score(y_test, bernoulli_y_predictions)
    print("\n\nBERNOULLI NAIVE-BAYES MODEL")
    print("Overall Accuracy: ", bernoulli_accuracy)
    print("Overall classification report:")
    print(classification_report(
        y_test, bernoulli_y_predictions, zero_division=0))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, bernoulli_y_predictions))

    # COMPLEMENT NB
    complement_model = ComplementNB()
    complement_model.fit(x_train, y_train)
    complement_y_predictions = complement_model.predict(x_test)
    complement_accuracy = accuracy_score(y_test, complement_y_predictions)
    print("\n\nCOMPLEMENT NAIVE-BAYES MODEL")
    print("Overall Accuracy: ", complement_accuracy)
    print("Overall classification report:")
    print(classification_report(
        y_test, complement_y_predictions, zero_division=0))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, complement_y_predictions))

    print("\n\nX_train shape:", np.array(x_train).shape)
    print("X_test shape:", np.array(x_test).shape)

    print("y_train shape:", np.array(y_train).shape)
    print("y_test shape:", np.array(y_test).shape)

    print(len(x_axis))

    for key in final_data:
        print("Country: " + key + " Probs: " + str(final_data[key] * 100))

    sorted_data = dict(sorted(final_data.items(), key=lambda item: item[1], reverse=True))
    print(sorted_data)
    x_vals = list(sorted_data.keys())[0:75]
    y_vals = list(sorted_data.values())[0:75]

    fig = px.bar(x=x_vals, y=y_vals, title='Probability of Country Winning Medal', labels={'x': 'Countries', 'y': 'Probability of Winning a Medal'})
    fig.show()


main()
