import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
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

def analyze_reviews(location, term, categories, dish_names):
    data = get_restaurants_and_reviews(location, term, categories)
    df = pd.DataFrame(data)
    df.drop_duplicates(subset=["review_text"], inplace=True)
    sid = SentimentIntensityAnalyzer()
    df["sentiment"] = df["review_text"].apply(lambda x: sid.polarity_scores(x)["compound"])
    for dish in dish_names:
        df[dish] = df["review_text"].apply(lambda x: 1 if dish.lower() in x.lower() else 0)
    df_dishes = df.groupby("restaurant_id")[dish_names].sum()
    df_dishes["best_selling_dish"] = df_dishes.idxmax(axis=1)
    return df, df_dishes.reset_index()[["restaurant_id", "best_selling_dish"]]

location = "Los Angeles, CA" #San Francisco, CA
term = "restaurants"
cuisine_dishes = {
    "Italian": ["pasta", "pizza", "lasagna", "risotto", "gnocchi", "spaghetti", "ravioli"],
    "Indian": ["biryani", "tandoori", "curry", "masala", "samosa", "dosa", "paneer"],
    "Mexican": ["tacos", "enchiladas", "burrito", "quesadilla", "guacamole", "salsa", "nachos"]
}

for cuisine, dish_names in cuisine_dishes.items():
    df, df_best_selling_dishes = analyze_reviews(location, term, cuisine, dish_names)
    df.to_csv(f"{cuisine}_reviews.csv", index=False)
    df_best_selling_dishes.to_csv(f"{cuisine}_best_selling_dishes.csv", index=False)

    # add restaurant name back into df_best_selling_dishes
    df_best_selling_dishes = df_best_selling_dishes.merge(df[["restaurant_id", "restaurant_name"]].drop_duplicates(), on="restaurant_id", how="left")

    # check if 'restaurant_name' in df_best_selling_dishes
    if 'restaurant_name' in df_best_selling_dishes.columns:
        best_selling_dishes_by_restaurant = df_best_selling_dishes.groupby("restaurant_name")["best_selling_dish"].first()
    else:
        print("The DataFrame doesn't have the 'restaurant_name' column.")

    if df_best_selling_dishes.empty:
        print("The DataFrame is empty.")
    else:
        best_selling_dishes_by_restaurant = df_best_selling_dishes.groupby("restaurant_name")["best_selling_dish"].first()

    plot_best_selling_dishes(best_selling_dishes_by_restaurant, f"{cuisine}_best_selling_dishes.png")
