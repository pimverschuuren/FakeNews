
import pandas as pd


# Print some general info on the dataset.
def print_dataset_info(dataset):

    print("Number of entries: "+str(len(dataset.index)))
    print("Col headers: "+str(list(dataset.columns)))
    print("First entry: ")
    print(dataset.iloc[0])
    print("First entry full text: ")
    print(dataset.iloc[0][1])
