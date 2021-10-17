
import pandas as pd
import torch
import itertools
import numpy as np
import nltk
import pickle

from nltk.stem import PorterStemmer

#from gensim.models import Word2Vec
#from sklearn.feature_extraction.text import TfidfVectorizer

# Print some general info on the dataset.
def print_dataset_info(dataset):

    print("Number of entries: "+str(len(dataset.index)))
    print("Col headers: "+str(list(dataset.columns)))
    print("First entry: ")
    print(dataset.iloc[0])
    print("First entry full text: ")
    print(dataset.iloc[0][1])

# Clean text in the column named text_head by:
# - Removing punctuation
def clean_text_data(dataframe, text_head):

    df_cleaned = dataframe.copy()

    # Only keep alphanumeric characters.
    df_cleaned[text_head] = df_cleaned[text_head].str.replace('[^\w\s]','')

    return df_cleaned

# Convert text in the column named text_head into
# a list of words i.e. a vocabulary that can be used for
# word to vector conversion. Only words that occur at least
# the min_freq times are included in the vocabulary.
def convert_df_to_vocab(dataframe, text_head, min_freq=10):

    df_vocab = dataframe.copy()

    # Only keep alphabetic characters in the text.
    df_vocab[text_head] = df_vocab[text_head].str.replace('[^a-zA-Z ]', '')

    # Make a list of words.
    word_list = list(filter(None, " ".join(list(itertools.chain(*df_vocab[text_head].str.split(' ')))).split(" ")))

    # Take the stems of words.
    ps = PorterStemmer()
    word_list = [ps.stem(word) for word in word_list]

    # Create a dataframe.
    df_words = pd.DataFrame(word_list,columns=['words'])

    # Get the frequencies of the words.
    df_word_freq = df_words.value_counts()

    word_list = []
    
    # Keep the words that occur at least min_freq times.
    for index, value in df_word_freq.items():
        if value > min_freq:
            word_list.append(str(index[0]))

    return word_list
