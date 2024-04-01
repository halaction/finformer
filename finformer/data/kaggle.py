import os
import requests
from dotenv import load_dotenv
import json
import pandas as pd


def get_data():

    load_dotenv()

    DATA_PATH = '../../data'
    os.makedirs(DATA_PATH, exist_ok=True)

    csv_path = os.path.join(DATA_PATH, 'analyst_ratings_processed.csv')
    df = pd.read_csv('csv_path')

    return df
