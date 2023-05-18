# Bestselling_prediction_project

Yelp Best-Selling Dish Analyzer
This Python application analyzes Yelp restaurant reviews to identify the best-selling dishes for restaurants in a specific location and for different cuisines.

Requirements
Python 3
Pandas
Numpy
Scikit-learn
NLTK
Matplotlib
Yelp Fusion API key
Data Source
This application uses the Yelp Fusion API to fetch restaurant data and reviews.

Overview of the Code
The code is broken down into different sections:

Importing necessary libraries: This includes pandas, numpy, requests, scikit-learn, nltk, and matplotlib.

Defining necessary functions: Functions for fetching Yelp data, analyzing reviews, and plotting the results are defined.

Main program execution: Here, we call the defined functions for a given location and a set of cuisines and their associated popular dishes. The results are saved as CSV files and a bar plot representing the best-selling dishes for each cuisine is created.

Detailed Breakdown of the Code
Importing necessary libraries: Libraries necessary for data manipulation, sentiment analysis, machine learning, API requests, and data visualization are imported.

Defining necessary functions:

plot_best_selling_dishes(data, output_file): This function takes the pandas Series of best-selling dishes and an output file name as input, and creates a bar plot showing the frequency of each dish. The plot is then saved as a PNG file.

get_restaurants_and_reviews(location, term, categories, limit=50): This function fetches the restaurant data and reviews from Yelp Fusion API. It takes location, term, categories, and limit (default is 50) as input, and returns a list of dictionaries, each containing a restaurant's id, name, review text, and rating.

analyze_reviews(location, term, categories, dish_names): This function analyzes the reviews fetched by get_restaurants_and_reviews function. It takes location, term, categories, and dish names as input, and returns a DataFrame containing the review data and a DataFrame containing the best-selling dish for each restaurant.

Main program execution: In the main execution of the program, we define the location, term, and cuisines with their associated popular dishes. We then call analyze_reviews function for each cuisine, and save the resulting DataFrames as CSV files. We also create a bar plot for the best-selling dishes using plot_best_selling_dishes function.

Execution
To run the script, follow the steps:

Replace the api_key placeholder with your Yelp Fusion API key.
Define your location, term, and cuisines with their associated popular dishes.
Run the script.
Output
The script will create a CSV file for each cuisine, containing the restaurant reviews and the best-selling dishes. It will also create a PNG file for each cuisine, containing a bar plot of the best-selling dishes.

Note
Make sure to handle API request errors and to respect Yelp's API usage guidelines when using this script. The current script does not implement error handling for simplicity. Also, sentiment analysis and dish popularity determination are based on simple methods and may not be accurate. For more precise results, consider using more sophisticated natural language processing methods.
