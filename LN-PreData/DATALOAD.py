import panda as pd
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
