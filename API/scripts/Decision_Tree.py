import pandas as pd
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=10):
        """Initialize the decision tree with a maximum depth."""
        self.max_depth = max_depth
        self.root = None

    def fit(self, data, labels):
        """Build the decision tree recursively."""
        self.root = self._fit(data, labels, depth=0)

    def _fit(self, data, labels, depth):
        """Recursively builds the decision tree."""
        if depth == self.max_depth or len(np.unique(labels)) == 1:
            # If pure or max depth reached, return a leaf node
            return DecisionTreeNode(value=self._most_common_label(labels))

        # Find best feature and threshold for splitting
        best_feature, best_threshold = self._best_split(data, labels)

        # Partition data based on the best threshold
        left_mask = data[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if np.all(left_mask) or np.all(right_mask):  # Prevent infinite splitting
            return DecisionTreeNode(value=self._most_common_label(labels))

        # Recursively build left and right subtrees
        left_child = self._fit(data[left_mask], labels[left_mask], depth + 1)
        right_child = self._fit(data[right_mask], labels[right_mask], depth + 1)

        return DecisionTreeNode(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    # To Label the Leaf Node
    def _most_common_label(self, labels):
        """Return the most common label in the dataset."""
        return Counter(labels).most_common(1)[0][0]

    def _entropy(self, labels):
        """Compute entropy of the labels using Counter for speed."""
        count = Counter(labels)
        total = len(labels)
        return -sum((freq / total) * np.log2(freq / total) for freq in count.values())

    def _compute_information_gain(self, data, feature_index, labels):
        """
        Computes information gain of a feature for a given dataset.
        """
        total_entropy = self._entropy(labels)

        # Calculate Information Gain without sorting
        left_mask = data[:, feature_index] <= data[:, feature_index].mean()
        right_mask = ~left_mask

        left_labels = labels[left_mask]
        right_labels = labels[right_mask]

        if len(left_labels) == 0 or len(right_labels) == 0:
            return 0, 0

        # Compute weighted entropy of the split
        weighted_entropy = (len(left_labels) / len(labels)) * self._entropy(left_labels) + \
                           (len(right_labels) / len(labels)) * self._entropy(right_labels)

        return total_entropy - weighted_entropy, data[:, feature_index].mean()

    def _best_split(self, data, labels):
        """Find the best feature and threshold for splitting."""
        n_features = data.shape[1]
        best_IG = 0
        best_threshold = 0
        best_feature_index = None

        # Loop through each feature to calculate information gain
        for i in range(n_features):
            info_gain, threshold = self._compute_information_gain(data, i, labels)

            if info_gain > best_IG:
                best_IG = info_gain
                best_feature_index = i
                best_threshold = threshold

        return best_feature_index, best_threshold

    def predict(self, data):
        """Predict labels for multiple samples."""
        return np.array([self._predict_sample(self.root, sample) for sample in data])

    def _predict_sample(self, node, sample):
        """Recursively predict the label for a single sample."""
        if node.value is not None:
            return node.value
        if sample[node.feature] <= node.threshold:
            return self._predict_sample(node.left, sample)
        else:
            return self._predict_sample(node.right, sample)

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
# tree.fit(features, labels)
# predictions = tree.predict(features)

# # Optionally, you can calculate accuracy or any other metric
# accuracy = np.mean(predictions == labels)
# print(f"Accuracy: {accuracy * 100:.2f}%")

