from helpers import *

# Choose the texts that are going to be used to
# source a vocabulary.
true_news_data = pd.read_csv("Datasets/True.csv")

# Create vocabulary in the form of a list.
true_vocab = convert_df_to_vocab(true_news_data, "text")

# Save the vocabulary to a file.
pickle.dump( true_vocab, open( "true_vocab.pickle", "wb" ) )
