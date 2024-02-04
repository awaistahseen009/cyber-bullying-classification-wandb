import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
from nltk.tokenize import word_tokenize , sent_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class DataPreprocessing:
    def __init__(self, data_frame : pd.DataFrame)->None:
        self.stemmer=PorterStemmer()
        self.data_frame=data_frame
        self.tokenizer= Tokenizer()
        self.sequences=None
        self.vocab_size=None
        self.max_len=40
        self.x=None
        self.y=None
        self.x_train=None
        self.y_train=None
        self.x_test=None
        self.y_test=None
    def rename_cols(self)->None:
        self.data_frame.columns= ['text','target']
    
    def get_dataframe(self)->pd.DataFrame:
        return self.data_frame

    def show_shape(self)->None:
        print(f'Shape of Dataframe is: {self.data_frame.shape}')

    def show_null_values(self)->None:
        print(f'Total num values are as follows: {self.data_frame.isna().sum()}')
    
    def remove_null_values(self)->None:
        self.data_frame.dropna(inplace=True)
        print('Null values are being removed')
    
    def show_duplicates(self)->None:
        print(f'Total duplicate values are : {self.data_frame.duplicated()}')
    
    def remove_duplicates(self)->None:
        self.data_frame.drop_duplicates(inplace=True)
        print('Duplicates values are being removed')

    def preprocess_text(self,text)->str:
        text = text.lower()
        text = word_tokenize(text)
        text = [word for word in text if not re.match(r'#\w+', word)]
        translator = str.maketrans('', '', string.punctuation)
        text = [word.translate(translator) for word in text]
        stop_words = set(stopwords.words('english'))
        text = [word for word in text if word not in stop_words]
        text=[self.stemmer.stem(word) for word in text]
        cleaned_text = " ".join(text)
        return cleaned_text
    
    def preprocess_dataframe(self ,separte_col=True):
        if separte_col:
            self.data_frame['clean_text']=self.data_frame['text'].apply(self.preprocess_text)
        else:
            self.data_frame['text']=self.data_frame['text'].apply(self.preprocess_text)
    
    def encode_target(self)->None:
        for i,value in enumerate(self.data_frame['target'].unique()): 
            self.data_frame['target'].replace({value:i},inplace=True)
    
    def prepare_training_data(self,separte_col=True):
        if separte_col:
            self.tokenizer.fit_on_texts(self.data_frame['clean_text'].values)
            self.sequences = self.tokenizer.texts_to_sequences(self.data_frame['clean_text'].values)
        else:
            self.tokenizer.fit_on_texts(self.data_frame['text'].values)
            self.sequences = self.tokenizer.texts_to_sequences(self.data_frame['text'].values)
        self.vocab_size=len(self.tokenizer.word_index)
        self.sequences=pad_sequences(self.sequences, maxlen=self.max_len, padding='post')
        self.x=self.sequences
        self.y=to_categorical(self.data_frame['target'].values)
        
    
    def split_data(self,test_size:np.float32):
        self.x_train, self.x_test,self.y_train,self.y_test=train_test_split(self.x , self.y , test_size=test_size)
        print(f'Traning has samples: {self.x_train.shape[0]} , Testing has samples: {self.x_test.shape[0]}')
    
    def get_split_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test

    def get_vocab_size(self):
        return len(self.tokenizer.word_index)
    
    def target_count(self):
        return len(self.data_frame['target'].unique())
    

        



        

            
