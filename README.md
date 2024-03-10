# Decision Tree Implementation 
This is a Python implementation of a decision tree for both regression and classification tasks. The code is designed to be modular, with separate classes for DecisionNode and LeafNode. It includes classes for both regression and classification trees, allowing for easy extension to different types of tasks.

## Usage

### Importing the Required Libraries

```python
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score
```

# Decision Tree Classes
### DecisionNode
Represents a decision node in the decision tree. <br>
Stores information about the split column, split value, and left/right child nodes.<br>
Has a predict method to make predictions based on the decision node.<br>
### LeafNode
Represents a leaf node in the decision tree. <br>
Stores information about the number of samples (n) and the prediction value.<br>
Has a predict method to return the prediction value.<br>
### Functions
#### gini(y)<br>
Calculates the Gini impurity score for values in y.<br>
Assumes y takes binary values (0 or 1).<br>
#### find_best_split(X, y, loss, min_samples_leaf)<br>
Finds the best split for the decision tree based on a given loss function.<br>
Uses a random subset of potential splits for efficiency.<br>
### DecisionTree621 Class
Main class for creating decision trees.<br>
Includes methods for fitting the tree to training data (fit), making predictions (predict), and creating a leaf node (create_leaf). <br>
### RegressionTree621 Class
Inherits from DecisionTree621.<br>
Specialized for regression tasks.<br>
Uses the variance as the loss function and calculates the R^2 score for predictions.<br>
#### Methods:
fit(X, y): Fits the regression tree to the training data.<br>
predict(X_test): Makes predictions for the given test data.<br>
score(X_test, y_test): Calculates the R^2 score for the predictions compared to the actual values.<br>
### ClassifierTree621 Class
Inherits from DecisionTree621.<br>
Specialized for classification tasks.<br>
Uses the Gini impurity as the loss function and calculates the accuracy score for predictions.<br>
#### Methods:
fit(X, y): Fits the classification tree to the training data.<br>
predict(X_test): Makes predictions for the given test data.<br>
score(X_test, y_test): Calculates the accuracy score for the predictions compared to the actual values.<br>
<br>
# Example usage for regression
```
reg_tree = RegressionTree621(min_samples_leaf=5)
reg_tree.fit(X_train, y_train)
predictions = reg_tree.predict(X_test)
r2 = reg_tree.score(X_test, y_test)
```
<br>

# Example usage for classification
```
class_tree = ClassifierTree621(min_samples_leaf=5)
class_tree.fit(X_train, y_train)
predictions = class_tree.predict(X_test)
accuracy = class_tree.score(X_test, y_test)
```

### Dependencies: <br>
- numpy
- scipy
- scikit-learn
### Installation
`pip install numpy scipy scikit-learn
`
