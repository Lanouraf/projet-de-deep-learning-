import pandas as pd
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd


def data_review():
    DATA_PATH = 'data/imdb_reviews.csv'
    if not Path(DATA_PATH).is_file():
        gdd.download_file_from_google_drive(
            file_id='1zfM5E6HvKIe7f3rEt1V2gBpw5QOSSKQz',
            dest_path=DATA_PATH,)
    DATA_PATH = 'data/imdb_reviews.csv'
    df = pd.read_csv(DATA_PATH)
    return df

def data_twitter():
    """Loads the Apple Twitter sentiment texts data from Kaggle.

    Returns
    -------
    df: pd.DataFrame
        The Apple Twitter sentiment texts data.

    Notes
    -----
    This is the dataset dowloaded from
        "https://www.kaggle.com/datasets/seriousran/appletwittersentimenttexts/download?datasetVersionNumber=1"

    """
    directory = "apple-twitter-sentiment-texts.csv"
    home_data = pd.read_csv(directory)
    return home_data

def data_apple():
    """Loads the Apple quality data from Kaggle.

    Returns
    -------
    df: pd.DataFrame
        apple quality dataset to do classification tasks.

    Notes
    -----
    This is the dataset dowloaded from
        "https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality"

    """
    directory = "apple_quality.csv"
    home_data = pd.read_csv(directory)
    return home_data
