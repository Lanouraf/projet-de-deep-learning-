import pandas as pd
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd

def data_review():
    """
    Loads the IMDb movie reviews data.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing IMDb movie reviews data.

    Notes
    -----
    This function checks if the IMDb movie reviews dataset exists locally. If not, it downloads the dataset from Google Drive.
    The dataset contains movie reviews along with their sentiments (positive or negative).

    """
    DATA_PATH = 'data/imdb_reviews.csv'
    # Check if the dataset exists locally
    if not Path(DATA_PATH).is_file():
        # Download the dataset from Google Drive if it doesn't exist
        gdd.download_file_from_google_drive(
            file_id='1zfM5E6HvKIe7f3rEt1V2gBpw5QOSSKQz',
            dest_path=DATA_PATH,
        )
    # Read the dataset into a DataFrame
    df = pd.read_csv(DATA_PATH)
    return df

def data_twitter():
    """
    Loads the Apple Twitter sentiment texts data.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing Apple Twitter sentiment texts data.

    Notes
    -----
    This function loads the Apple Twitter sentiment texts dataset from Kaggle.
    The dataset consists of tweets related to Apple, along with their sentiment labels.

    """
    directory = "apple-twitter-sentiment-texts.csv"
    # Read the dataset into a DataFrame
    home_data = pd.read_csv(directory)
    return home_data

def data_apple():
    """
    Loads the Apple quality data.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing Apple quality dataset for classification tasks.

    Notes
    -----
    This function loads the Apple quality dataset from Kaggle.
    The dataset contains quality-related data about Apple products, suitable for classification tasks.

    """
    directory = "apple_quality.csv"
    # Read the dataset into a DataFrame
    home_data = pd.read_csv(directory)
    return home_data
