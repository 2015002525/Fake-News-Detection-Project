import pandas as pd
from nltk import word_tokenize


class DataPreProcessor:
    DATA_PATH = ""

    def __init__(self, PATH):
        self.DATA_PATH = PATH

    def LoadData(self):
        print("h")
        self.df = pd.read_excel(self.DATA_PATH)
        print(self.df)
        return self.df

    def PrepareData(self):
        print("d5lt prepareData")
        try:
            self.df = self.df.drop(['id'], axis=1)  # drop id
            self.df = self.df.drop(['author'], axis=1)  # drop author
            self.df['content'] = self.df['title'].astype(str) + self.df['text'] # merge title with text into new column called content
            self.df = self.df.drop(['title'], axis=1)  # drop title column
            self.df = self.df.drop(['text'], axis=1)  # drop text column
            self.df.to_csv("out2.csv") #save new dataframe as out.csv
            print("Done")
            print(self.df)
        except:
            print("the data is already preprocessed")


    def PreProcess(self):
        label = []
        content = []
        for j in range(0, len(self.df)):
            content_data = str(self.df['content'][j]).lower()
            label_data = self.df.iloc['label'][j]
            tokenized_words = word_tokenize(content_data)
            label.append(label_data)
            content.append(tokenized_words)
            tokenized_words = []
        modified_dataFrame = pd.DataFrame({'label': label, 'content': content})
        modified_dataFrame.to_csv("prprprprp.csv")


#if __name__ == '__main__':
   # DataPreProcessor("C:\\Users\\Ahmed\\OneDrive\\Desktop\\Fake-News-Detection-master\\FakeNewsDataset\\Train.csv")
    #DataPreProcessor("C:\\Users\\Ahmed\\OneDrive\\Desktop\\Fake-News-Detection-master\\BOOK.csv")