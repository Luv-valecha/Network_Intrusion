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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype
import joblib  # Add this import for saving the encoders

def encode_columns(df):
    cat_var = [col for col in df.columns if not is_numeric_dtype(df[col])]
    mappings = {}  # Dictionary to store mappings for each column

    # Create the directory to save encoders if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    encoders_dir = os.path.join(current_dir, "..", "model", "saved_models")
    os.makedirs(encoders_dir, exist_ok=True)

    for x in cat_var:
        label_encoder = LabelEncoder()  # Create a new encoder for each column
        df[x] = label_encoder.fit_transform(df[x])
        mappings[x] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        joblib.dump(label_encoder, os.path.join(encoders_dir, f"{x}_encoder.pkl"))

    return df  # Ensure df is returned after transformation

def preprocess(datapath,targetdatapath,k,test_size = 0.2):
    """
    This function preprocesses the data and saves it to the targetdatapath
    parameters:
        datapath: The path to the raw data (str)
        targetdatapath: The path to save the processed data (str)
        k: The number of features to select (int)
    """

    # datapath
    trainpath = os.path.join(datapath,"Train_data.csv")

    # Load the data
    train_data = pd.read_csv(trainpath)

    # Encode the categorical columns
    encode_columns(train_data)

    # labesl
    labels = train_data['class']
    train_data = train_data.drop(columns=['class'])

    # Drop the columns with more than 50% missing values
    train_data = train_data.dropna(thresh=0.5*train_data.shape[0],axis=1)

    # Drop the columns with only one unique value
    train_data = train_data.loc[:,train_data.apply(pd.Series.nunique) != 1]

    # Drop the columns with more than 90% zeros
    train_data = train_data.loc[:,(train_data==0).mean() < 0.9]

    # split the data in train test
    X_train, X_test, y_train, y_test = train_test_split(train_data,labels, random_state= 33,test_size = test_size,shuffle=True)

    # Select only the best features
    train_data,test_data = GetKbestfeatures(X_train, y_train ,X_test, k)

    # Create an independent copy
    train_data = train_data.copy()  
    test_data = test_data.copy()
    
    # add the labels back
    train_data['class'] = y_train
    test_data["class"] = y_test

    # Save the processed data
    train_data.to_csv(os.path.join(targetdatapath,"train_data.csv"),index=False)
    test_data.to_csv(os.path.join(targetdatapath,"test_data.csv"),index=False)

    
    
if __name__ == "__main__":

    # Get the directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the datapath relative to the script file location
    datapath = os.path.join(current_dir, "..", "data", "raw")
    # create the processed folder if it does not exist
    os.makedirs(os.path.join(current_dir, "..", "data", "processed"), exist_ok=True)
    targetdatapath = os.path.join(current_dir, "..", "data", "processed")

    k = 10
    preprocess(datapath, targetdatapath, k,test_size=0.2)