from sklearn.feature_selection import SelectKBest,mutual_info_classif
import pandas as pd 


def GetKbestfeatures(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, k: int) -> list:
        """
        This function selects the best features from the dataset using mutual information
        parameters:
        X_train: The training set (DataFrame)
        y_train: The training labels (Series)
        X_test: The test set (DataFrame)
        k: The number of features to select (int)
        returns:
        X_train: The training set with only the best features (DataFrame)
        X_test: The test set with only the best features (DataFrame)
        """
        # Select the best features
        selector = SelectKBest(mutual_info_classif, k=k)

        # Fit the selector
        selector.fit(X_train, y_train)

        # Get the selected features
        selected_features = X_train.columns[selector.get_support()]

        # Select only the best features
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

        # Return the data
        return X_train, X_test