import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import nltk
import matplotlib.pyplot as plt
nltk.download('vader_lexicon')

def plot_best_selling_dishes(data, output_file):
    plt.figure(figsize=(10, 6))
    data.value_counts().plot(kind="bar")
    plt.xlabel("Dishes")
    plt.ylabel("Frequency")
    plt.title("Best-selling dishes")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


api_key = "KSDjeO2X3PwY2c_zCYc_fomHMHsQje3cb8NybeNMspPgJ_7nbW2UQRRPy5gWxchiXBoetIyLYAY_kZTQHBJ6RZutdPXCvAFB8rfaT-OfgLjJHuVcB_2wj_GpGvVXZHYx"
headers = {'Authorization': 'Bearer %s' % api_key}
url = "https://api.yelp.com/v3/businesses/search"

def get_restaurants_and_reviews(location, term, categories, limit=50):
    data = []
    for offset in range(0, 550, limit):
        params = {
            "location": location,
            "term": term,
            "categories": categories,
            "limit": limit,
            "offset": offset,
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            restaurants = response.json()["businesses"]
            for restaurant in restaurants:
                restaurant_id = restaurant["id"]
                restaurant_name = restaurant["name"]
                review_response = requests.get(f"https://api.yelp.com/v3/businesses/{restaurant_id}/reviews", headers=headers)
                if review_response.status_code == 200:
                    reviews = review_response.json()["reviews"]
                    for review in reviews:
                        data.append({
                            "restaurant_id": restaurant_id,
                            "restaurant_name": restaurant_name,
                            "review_text": review["text"],
                            "rating": review["rating"],
                        })
        else:
            print(f"Request failed with status code {response.status_code}")
    return data

location = "San Francisco, CA" # "New York, NY","Los Angeles, CA",m"Chicago, IL", "Houston, TX", "Philadelphia, PA"
term = "restaurants"
categories = "italian" # or italian or mexican

data = get_restaurants_and_reviews(location, term, categories)

df = pd.DataFrame(data)
df.drop_duplicates(subset=["review_text"], inplace=True)

sid = SentimentIntensityAnalyzer()
df["sentiment"] = df["review_text"].apply(lambda x: sid.polarity_scores(x)["compound"])

dish_names = ["pasta", "pizza", "lasagna", "risotto", "gnocchi", "spaghetti", "ravioli"]


for dish in dish_names:
    df[dish] = df["review_text"].apply(lambda x: 1 if dish.lower() in x.lower() else 0)

df_dishes = df.groupby("restaurant_id")[dish_names].sum()
df_dishes["best_selling_dish"] = df_dishes.idxmax(axis=1)

df_best_selling_dishes = df_dishes.reset_index()[["restaurant_id", "best_selling_dish"]]
df_final = df.merge(df_best_selling_dishes, on="restaurant_id")
df_final.to_csv("best_selling_dishes.csv", index=False)

# Prepare the dataset for training the model
df_dish_labels = pd.get_dummies(df[dish_names])
df = pd.concat([df, df_dish_labels], axis=1)

# Split the data into training and testing sets
X = df[["review_text", "sentiment"]]
y = df[dish_names]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the review text
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train["review_text"])
X_test_vectorized = vectorizer.transform(X_test["review_text"])

# Add the sentiment score as an additional feature
X_train_vectorized = np.hstack((X_train_vectorized.toarray(), X_train[["sentiment"]].to_numpy()))
X_test_vectorized = np.hstack((X_test_vectorized.toarray(), X_test[["sentiment"]].to_numpy()))

# Train a LogisticRegression classifier with MultiOutputClassifier
clf = MultiOutputClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = clf.predict(X_test_vectorized)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#Group the data by restaurant and find the best-selling dish for each restaurant
best_selling_dishes_by_restaurant = df_final.groupby("restaurant_name")["best_selling_dish"].first()

# Plot the best-selling dishes and save the output as a PNG image
plot_best_selling_dishes(best_selling_dishes_by_restaurant, "best_selling_dishes.png")
