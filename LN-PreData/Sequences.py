import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, Dataset
from DATALOAD import data_twitter
home_data=data_twitter()
class Sequences(Dataset):
        def __init__(self, home_data, vectorizer):
            self.vectorizer = vectorizer
            list_texts = home_data.text.tolist()
            self.sequences = self.vectorizer.transform(list_texts)
            self.sentiments = home_data.sentiment.tolist()

        def __getitem__(self, i):
            sequence_i = self.sequences[i]
            sentiment_i = self.sentiments[i]
            return sequence_i.toarray(), sentiment_i

        def __len__(self):
            return self.sequences.shape[0]


vectorizer = CountVectorizer(stop_words="english", max_df=0.99, min_df=0.005)
list_texts = home_data.text.tolist()
vectorizer.fit(list_texts)

train_home_data = home_data.iloc[:int(0.7*home_data.shape[0])]
test_home_data = home_data.iloc[int(0.7*home_data.shape[0]):]

train_dataset = Sequences(train_home_data, vectorizer)
test_dataset = Sequences(test_home_data, vectorizer)
train_loader = DataLoader(dataset=train_dataset, batch_size=4096)
test_loader = DataLoader(dataset=test_dataset, batch_size=4096)
print(list_texts)
