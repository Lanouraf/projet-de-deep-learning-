"""
This file contains functions and classes for loading and preparing data for sentiment classification using Layer Normalization.

Functions:
    - data_review(): Loads the home data containing reviews and labels.
    - Sequences(Dataset): Custom dataset class for sequences, used for tokenizing text data.
    - prep_data(): Prepares the data for training and testing by creating datasets and data loaders.
"""
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, Dataset
from LN_dataload import data_review

# Load the home data
home_data = data_review()

# Define a custom dataset class for sequences
class Sequences(Dataset):
    def __init__(self, home_data, vectorizer):
        """
        Initialize the Sequences dataset.

        Args:
            home_data (DataFrame): DataFrame containing the home data.
            vectorizer (CountVectorizer): CountVectorizer object for tokenizing the text data.
        """
        self.vectorizer = vectorizer
        list_reviews = home_data.review.tolist()  # Create a list of reviews
        self.sequences = self.vectorizer.transform(list_reviews)  # Generate sequences using the vectorizer
        self.labels = home_data.label.tolist()  # Convert labels to a list

    def __getitem__(self, i):
        """
        Get a sequence and its corresponding label at index i.

        Args:
            i (int): Index of the sequence.

        Returns:
            tuple: Tuple containing the sequence and its label.
        """
        sequence_i = self.sequences[i, :]  # Select the sequence at index i
        label_i = self.labels[i]  # Select the label at index i
        return sequence_i.toarray(), label_i

    def __len__(self):
        """
        Get the total number of sequences in the dataset.

        Returns:
            int: Total number of sequences.
        """
        return self.sequences.shape[0]

def prep_data():
    """
    Prepare the data for training and testing.

    Returns:
        tuple: Tuple containing train and test datasets and loaders.
    """
    # Initialize a CountVectorizer object
    vectorizer = CountVectorizer(stop_words="english", max_df=0.99, min_df=0.005)
    list_texts = home_data.review.tolist()
    vectorizer.fit(list_texts)  # Fit the vectorizer on the text data

    # Split the data into train and test sets
    train_home_data = home_data.iloc[:int(0.7 * home_data.shape[0])]
    test_home_data = home_data.iloc[int(0.7 * home_data.shape[0]):]

    # Create datasets and data loaders
    train_dataset = Sequences(train_home_data, vectorizer)
    test_dataset = Sequences(test_home_data, vectorizer)
    train_loader = DataLoader(dataset=train_dataset, batch_size=4096)
    test_loader = DataLoader(dataset=test_dataset, batch_size=4096)

    return train_dataset, train_loader, test_dataset, test_loader
