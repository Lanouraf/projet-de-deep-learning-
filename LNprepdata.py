import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, Dataset
from DATALOAD import data_review
home_data=data_review()
home_data.head()
class Sequences(Dataset):
    def __init__(self,home_data, vectorizer):
        ### TODO create tokens for your dataset
        self.vectorizer = vectorizer
        # Create a list containing all the reviews from df
        list_reviews = home_data.review.tolist()
        # Generate your sequences using your vectorizer
        self.sequences = self.vectorizer.transform(list_reviews)
        ###
        # We convert the labels to a list of labels (before it was within a dataframe)
        self.labels = home_data.label.tolist()

    def __getitem__(self, i):
        ### TODO: self.sequences is a sparse matrix, where the rows contain the reviews and the columns the unique words within the dataset
        # Select the sequence at the index i
        sequence_i = self.sequences[i, :]
        # Select the label at the index i
        label_i = self.labels[i]
        ###

        # We return here the sequence and the label at the index i. We convert the sparse matrix to a numpy array.
        return sequence_i.toarray(), label_i

    def __len__(self):
        return self.sequences.shape[0]

def prep_data():
     vectorizer = CountVectorizer(stop_words="english", max_df=0.99, min_df=0.005)
     list_texts = home_data.review.tolist()
     vectorizer.fit(list_texts)
     train_home_data = home_data.iloc[:int(0.7*home_data.shape[0])]
     test_home_data = home_data.iloc[int(0.7*home_data.shape[0]):]
     train_dataset = Sequences(train_home_data, vectorizer)
     test_dataset = Sequences(test_home_data, vectorizer)
     train_loader = DataLoader(dataset=train_dataset, batch_size=4096)
     test_loader = DataLoader(dataset=test_dataset, batch_size=4096)
     return train_dataset,train_loader,test_dataset,test_loader
