import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
import re
import pycld2 as cld2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
nlp = spacy.load('en_core_web_sm', disable = ['parser','ner'])
nltk_stopwords = stopwords.words('english')

def tokenize_nltk(text):
    text = url_regex.sub(' ', text.lower())
    words = word_tokenize(re.sub(r'[\W\d_]',' ', text))
    words = [w for w in words if w not in nltk_stopwords]
    for pos in 'nvars':
        words = [WordNetLemmatizer().lemmatize(w, pos=pos) for w in words]
    return words

def tokenize_spacy(text):
    text = url_regex.sub(' ', text.lower())
    words = [token.lemma_ for token in nlp(text) \
                 if token.is_alpha and ~token.is_stop]
    return words

def prepare_df(df, tokenizer='spacy', test_size=0.2, random_state=42):
    if tokenizer is not None: # first time prepare
        df['related'] = np.where(df['related']==1, 1, 0)
        df.drop(columns=['child_alone'], inplace=True)
    if tokenizer == 'spacy':
        df['message'] = df['message'].apply(tokenize_spacy).apply(' '.join)
    elif tokenizer == 'nltk':
        df['message'] = df['message'].apply(tokenize_nltk).apply(' '.join)
    else:
        pass
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    cols = df.drop(columns=['message','original','id','genre']).columns
    return df_train, df_test, cols

class DetectLang(BaseEstimator, TransformerMixin):
    '''
      detect language for given df['original']
      the dummies are harwired to the top languages used
    '''
    def __init__(self, max_langs=5):
        self.max_langs = max_langs
        self.encoder = OneHotEncoder(max_categories=max_langs,
                                     handle_unknown='infrequent_if_exist')
    
    def _detect(self, X):
        # the return needs to be in [n_samples, n_features]
        return [[cld2.detect(x)[2][0][0]] if isinstance(x, str) else ['ENGLISH'] for x in X]
    
    def fit(self, X, y=None):
        print('restarting detectlang')
        langs = self._detect(X)
        self.encoder.fit(langs)  #this actually not going to do anything
        return self
    
    def transform(self, X):
        langs = self._detect(X)
        return self.encoder.transform(langs)

class RandomForestClassifier_wrapper(RandomForestClassifier):
    ''' add the transform method '''
    def transform(self, X):
        return (self.predict_proba(X))[:,1].reshape(-1,1)

class ColumnTransformer_wrapper(ColumnTransformer):
    ''' 
      override the fit, tranform and predict so it is at each transformer
      the current version only deals with all columns in X
    '''
    def fit(self, X, Y):
        for icol, (name, transformer, cols) in enumerate(self.transformers):
            transformer.fit(X, Y[:,icol])
        return self
    
    def transform(self, X):
        X_output = []
        for _, transformer, cols in self.transformers:
            X_output.append(transformer.transform(X))
        return np.hstack(X_output)
    
    def fit_transform(self, X, Y):
        return self.fit(X, Y).transform(X)
    
    def predict(self, X):
        output = []
        for _, transformer, cols in self.transformers:
            output.append(transformer.predict(X).reshape(-1,1))
        return np.hstack(output)
