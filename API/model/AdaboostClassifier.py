import os,sys
import joblib, pickle
import json
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Add the parent directory so that the 'scripts' folder is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import ModelEvaluator 

class adaboostClassifier:
    def __init__(self,saved_model_path : str = r"API\model\saved_models"):
        self.saved_model_path = saved_model_path
        self.classifier = None
    
    def train(self,estimator,data_path,n_estimators=50):
        df=pd.read_csv(data_path)
        x_train=df.drop(['class'],axis=1)
        y_train=df['class']
        self.classifier=AdaBoostClassifier(estimator=estimator,
                                           n_estimators=n_estimators,
                                           learning_rate=1.0,
                                           algorithm='SAMME',
                                           random_state=42)
        self.classifier.fit(x_train,y_train)

    def predict(self,x_test : pd.DataFrame ):
        if(self.classifier == None):
            raise ValueError("model is not initialised")
        return self.classifier.predict(x_test)
        
    def save_model(self, save_path: str = r"API\model\saved_models\voting_model.pkl"):
        """Save the trained model to a file."""
        if self.classifier is None:
            raise ValueError("Model is not trained. Call train() before saving.")
    
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path,"wb") as f:
            pickle.dump(self.classifier,f)
            print("Model saved")        

if __name__ == "__main__":
    # Set training dataset path
    train_dataset_path = r"API\data\processed\train_data.csv" 
    test_dataset_path = r"API\data\processed\test_data.csv"
    # Initialize classifier with training data
    classifier = adaboostClassifier()
    estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
    classifier.train(estimator,train_dataset_path,50)

    # Save the trained model
    model_save_path = r"API\model\saved_models\adaboost_model.pkl"
    classifier.save_model(model_save_path)

    # evaluate
    evaluater = ModelEvaluator("adaboost_model.pkl")
    evaluater.evaluate()