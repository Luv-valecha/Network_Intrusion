import pandas as pd
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_leaf=1, min_information_gain=0.0, numb_of_features_splitting=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.root = None
        self.unique_labels = None

    def train(self, data, labels):
        """Build the decision tree recursively."""
        self.unique_labels = np.unique(labels)  # Store unique labels for probability computation
        self.root = self._train(data, labels, depth=0)

    def _train(self, data, labels, depth):
        """Recursively builds the decision tree."""
        if depth == self.max_depth or len(np.unique(labels)) == 1 or len(labels) < self.min_samples_leaf:
            return DecisionTreeNode(value=self._most_common_label(labels))

        best_feature, best_threshold = self._best_split(data, labels)
        
        if best_feature is None:  # No valid split found
            return DecisionTreeNode(value=self._most_common_label(labels))

        left_mask = data[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if np.all(left_mask) or np.all(right_mask):
            return DecisionTreeNode(value=self._most_common_label(labels))

        left_child = self._train(data[left_mask], labels[left_mask], depth + 1)
        right_child = self._train(data[right_mask], labels[right_mask], depth + 1)

        return DecisionTreeNode(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _most_common_label(self, labels):
        """Return the most common label in the dataset."""
        return Counter(labels).most_common(1)[0][0]

    def _entropy(self, labels):
        """Compute entropy of the labels."""
        count = Counter(labels)
        total = len(labels)
        return -sum((freq / total) * np.log2(freq / total) for freq in count.values())

    def _compute_information_gain(self, data, feature_index, labels):
        """Computes information gain of a feature for a given dataset."""
        total_entropy = self._entropy(labels)

        left_mask = data[:, feature_index] <= data[:, feature_index].mean()
        right_mask = ~left_mask

        left_labels = labels[left_mask]
        right_labels = labels[right_mask]

        if len(left_labels) == 0 or len(right_labels) == 0:
            return 0, 0

        weighted_entropy = (len(left_labels) / len(labels)) * self._entropy(left_labels) + \
                           (len(right_labels) / len(labels)) * self._entropy(right_labels)

        return total_entropy - weighted_entropy, data[:, feature_index].mean()

    def _best_split(self, data, labels):
        """Find the best feature and threshold for splitting."""
        n_features = data.shape[1]
        best_IG = 0
        best_threshold = None
        best_feature_index = None

        if self.numb_of_features_splitting:
            feature_indices = np.random.choice(n_features, self.numb_of_features_splitting, replace=False)
        else:
            feature_indices = range(n_features)

        for i in feature_indices:
            info_gain, threshold = self._compute_information_gain(data, i, labels)

            if info_gain > best_IG:
                best_IG = info_gain
                best_feature_index = i
                best_threshold = threshold

        if best_IG < self.min_information_gain:
            return None, None

        return best_feature_index, best_threshold

    def predict_proba(self, X_set: np.array) -> np.array:
        """Returns the predicted probabilities for a given dataset."""
        pred_probs = np.array([self._predict_prob_one_sample(x) for x in X_set])
        return pred_probs

    def predict(self, data):
        """Returns the predicted labels for a given dataset."""
        pred_probs = self.predict_proba(data)
        preds = np.argmax(pred_probs, axis=1)
        return preds  

    def _predict_prob_one_sample(self, X):
        """Returns prediction probabilities for a single sample."""
        node = self.root

        while node and node.value is None:
            if X[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        if node.value is not None:
            return self._get_probabilities(node.value)
        else:
            return np.zeros(len(set(self.root.value)))  # Fallback if tree is not trained properly

    def _get_probabilities(self, label):
        """Returns a probability distribution for the predicted class."""
        if self.unique_labels is None:
            raise ValueError("Tree has not been trained yet!")

        probs = np.zeros(len(self.unique_labels))
        label_index = np.where(self.unique_labels == label)[0][0]  # Find the index of the label
        probs[label_index] = 1  # Assign probability 1 to the predicted label
        return probs

class DecisionTreeNode:
    """Class representing a node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value






# # Assuming the CSV file is in the folder './API/data/raw/archive/Train_data.csv'
# csv_path_train = './API/data/raw/archive/Train_data.csv'
# # train_and_predict(csv_path)
# features, labels = load_data(csv_path_train)      
# print(features.shape)

# #Set The MAX_depth Before Training and Inside Load Data total Rows If for testing
# tree = DecisionTree(max_depth=3)
# tree.train(features, labels)
# predictions = tree.predict(features)

# # Optionally, you can calculate accuracy or any other metric
# accuracy = np.mean(predictions == labels)
# print(f"Accuracy: {accuracy * 100:.2f}%")

