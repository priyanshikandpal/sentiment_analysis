import pandas as pd
import os
import pickle

class TwitterData:
    def __init__(self):
        self.data_file = 'data/Twitter_Data.csv'  # Path to your local dataset

    def getTwitterData(self, keyword, time):
        # Load tweets from the local CSV file
        if not os.path.exists(self.data_file):
            print(f"Error: The data file {self.data_file} does not exist.")
            return []

        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.data_file)

        # Filter tweets based on the keyword
        tweets = df[df['text'].str.contains(keyword, case=False, na=False)]

        # If time is specified as 'today', filter for today's date
        if time == 'today':
            today = pd.to_datetime("today").normalize()
            tweets = tweets[tweets['created_at'].dt.date == today.date()]

        # Return the filtered tweets as a list of texts
        return tweets['text'].tolist()

# Note: The parse_config and oauth_req methods are removed since they are no longer needed.