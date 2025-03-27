from Decision_Tree import DecisionTree
import numpy as np
import pandas as pd
import random
import pickle
import os, sys

# Add the parent directory so that the 'scripts' folder is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import ModelEvaluator 


class RandomForestClassifier:

    # constructor
    def __init__(self, n_learners=10, max_depth=5, min_samples_leaf=1, min_information_gain=0.0,
                 numb_of_features_splitting=None, bootstrap_sample_size=None):
        self.n_base_learner = n_learners
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.bootstrap_sample_size = bootstrap_sample_size

    # fucntion to create bootstrap samples for the decision trees
    def create_bootstrap_samples(self,X,Y):
        X_bootstrap=[]
        Y_bootstrap=[]

        # create bootstrap samples for every learner
        for i in range(self.n_base_learner):

            # if bootstrap sample size is not given by the user take it as number of rows
            if not self.bootstrap_sample_size:
                self.bootstrap_sample_size = X.shape[0]

            # randomly select teh required number of data rows
            bootstrap_idx=np.random.choice(X.shape[0], size=self.bootstrap_sample_size, replace=True)
            X_bootstrap.append(X[bootstrap_idx])
            Y_bootstrap.append(Y[bootstrap_idx])

        return X_bootstrap, Y_bootstrap
    
    # model training
    def train(self, X_train: np.array, Y_train: np.array):
        
        # create bootstrap samples
        bootstrap_samples_X, bootstrap_samples_Y = self.create_bootstrap_samples(X_train, Y_train)

        # list initialization to store decision tree learners
        self.base_learner_list = []

        # train the base decision trees
        for base_learner_idx in range(self.n_base_learner):
            base_learner = DecisionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                        min_information_gain=self.min_information_gain, 
                                        numb_of_features_splitting=self.numb_of_features_splitting)
            
            base_learner.train(bootstrap_samples_X[base_learner_idx], bootstrap_samples_Y[base_learner_idx])
            self.base_learner_list.append(base_learner)

        # Calculate feature importance
        # self.feature_importances = self.calculate_rf_feature_importance(self.base_learner_list)

    # function to predict decision for base learners
    def base_learners_predict(self,  X_set: np.array) -> list:

        pred_prob_list = []

        # add predicted probability from every decision tree in the list
        for base_learner in self.base_learner_list:
            pred_prob_list.append(base_learner.predict_proba(X_set))

        return pred_prob_list

    # predict function for random forest
    def probability_predict(self, X_set: np.array) -> list:

        pred_probs = []
        base_learners_pred_probs = self.base_learners_predict(X_set)

        # averaging the predicted probability from each decision tree 
        for i in range(X_set.shape[0]):
            base_learner_probs_for_obs = [a[i] for a in base_learners_pred_probs]
            obs_average_pred_probs = np.mean(base_learner_probs_for_obs, axis=0)
            pred_probs.append(obs_average_pred_probs)

        return pred_probs

    # main predict function to give the predicted label
    def predict(self, X_set: np.array) -> np.array:

        pred_probs = self.probability_predict(X_set)

        # highest probability label is predicted
        preds = np.argmax(pred_probs, axis=1)
        
        return preds

    # def calculate_rf_feature_importance(self, base_learners):
    #     """Calcalates the average feature importance of the base learners"""
    #     feature_importance_dict_list = []
    #     for base_learner in base_learners:
    #         feature_importance_dict_list.append(base_learner.feature_importances)

    #     feature_importance_list = [list(x.values()) for x in feature_importance_dict_list]
    #     average_feature_importance = np.mean(feature_importance_list, axis=0)

    #     return average_feature_importance

    # function to save the model
    def save_model(self, filename):
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
            print(f"Model saved to {filename}")
        except (OSError, IOError) as e:
            print(f"Error saving model: {e}")

    # function to load the saved model
    @staticmethod
    def load_model(filename):
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found.")
            return None
        try:
            with open(filename, 'rb') as file:
                model = pickle.load(file)
            print(f"Model loaded from {filename}")
            return model
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error loading model: {e}")
            return None

if __name__ == "__main__":
    # Set training dataset path
    train_dataset_path = r"API\data\processed\train_data.csv" 
    test_dataset_path = r"API\data\processed\test_data.csv"

    # Read the datapaths
    train = pd.read_csv(train_dataset_path)
    train_label = train["class"]
    train.drop('class',inplace=True,axis=1)

    test = pd.read_csv(test_dataset_path)
    test_label = test["class"]
    test.drop('class',inplace=True,axis=1)

    # Initialize classifier with training data
    classifier = RandomForestClassifier()
    classifier.train(train,train_label)

    # Save the trained model
    model_save_path = r"API\model\saved_models\rf_model.pkl"
    classifier.save_model(model_save_path)

    # evaluate
    evaluater = ModelEvaluator("rf_model.pkl")
    evaluater.evaluate()