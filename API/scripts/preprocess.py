"""Here we will load raw data from the data folder "../API/data/raw" and will 
save it to the data folder "../API/data/processed" after preprocessing it.

The preprocessing steps include:
1. Load the raw data
2. Drop the columns with more than 50% missing values
3. Drop the columns with only one unique value
4. Drop the columns with more than 90% zeros
5. Normalise the data
6. Select only top k features using mutual information
7. Save the processed data to the data folder "../API/data/processed"

"""

# Imports 
import pandas as pd
from PCA import GetKbestfeatures
import os

from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype

def encode_columns(df):
    cat_var = []
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            cat_var.append(col)
    
    label_encoder = LabelEncoder()
    mappings = {}  # Dictionary to store mappings for each column

    for x in cat_var:
        df[x] = label_encoder.fit_transform(df[x])
        # Store the mapping for this column
        mappings[x] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

def preprocess(datapath,targetdatapath,k):
    """
    This function preprocesses the data and saves it to the targetdatapath
    parameters:
        datapath: The path to the raw data (str)
        targetdatapath: The path to save the processed data (str)
        k: The number of features to select (int)
    """

    # datapaths
    testpath = os.path.join(datapath,"Test_data.csv")
    trainpath = os.path.join(datapath,"Train_data.csv")

    # Load the data
    test_data = pd.read_csv(testpath)
    train_data = pd.read_csv(trainpath)

    # Encode the categorical columns
    encode_columns(train_data)
    encode_columns(test_data)

    # labesl
    labels = train_data['class']
    train_data = train_data.drop(columns=['class'])

    # Drop the columns with more than 50% missing values
    train_data = train_data.dropna(thresh=0.5*train_data.shape[0],axis=1)
    test_data = test_data[train_data.columns]

    # Drop the columns with only one unique value
    train_data = train_data.loc[:,train_data.apply(pd.Series.nunique) != 1]
    test_data = test_data[train_data.columns]

    # Drop the columns with more than 90% zeros
    train_data = train_data.loc[:,(train_data==0).mean() < 0.9]
    test_data = test_data[train_data.columns]

    # Select only the best features
    train_data,test_data = GetKbestfeatures(train_data, labels ,test_data, k)

    # Normalise the data
    train_data = (train_data - train_data.mean())/train_data.std()
    test_data = (test_data - test_data.mean())/test_data.std()

    # add the labels back
    train_data['class'] = labels

    # Save the processed data
    train_data.to_csv(os.path.join(targetdatapath,"train_data.csv"),index=False)
    test_data.to_csv(os.path.join(targetdatapath,"test_data.csv"),index=False)

    
    
if __name__ == "__main__":

    # Get the directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the datapath relative to the script file location
    datapath = os.path.join(current_dir, "..", "data", "raw")
    targetdatapath = os.path.join(current_dir, "..", "data", "processed")

    k = 10
    preprocess(datapath, targetdatapath, k)