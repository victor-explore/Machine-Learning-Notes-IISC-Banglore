# Decision Trees

## How to descion tree works

- At each node, a question is asked about the data that splits the data into two or more non-overlapping subsets.
$$
\text{Question: } x_j \leq \theta 
$$
then the data is split into two subsets, one where $x_j \leq \theta$ and one where $x_j > \theta$.


![Decision Tree Diagram](Images\desciontrees.JPG)   

- The process is repeated till we reach leaf node, that classifies the datapoint to a region of the feature space.
![alt text](Images/image.png)

- We can do regression and classification of the test datapoint based on the trainig datapoints that lie in the region.

## Growing a Decision Tree

Choose a data dimension $j$ and a threshold $\theta$ to split the data that minimises a metric eg  minimize gini impurity for classification or minimize mean squared error for regression. Do this recursively.

## Gini Impurity
Gini Impurity is a measure of impurity or disorder in a set of data points. It's used to determine the quality of a split in a decision tree. The goal is to minimize the Gini Impurity when growing the tree.

The Gini Impurity is calculated as:

$$ G(set) = \sum_{i=1}^K \sum_{j \neq i} p_i p_j $$

$$ G(set) = 1 - \sum_{i=1}^K p_i^2 $$

Where:
- $K$ is the total number of classes
- $p_i$ is the probability of picking a data point with class $i$($p_j$ also defined similarly) if you randomly choose from the set
$$ p_i = \frac{n_i}{N} $$
Where:
- $n_i$ is the number of datapoints of class $i$ in the set
- $N$ is the total number of datapoints in the set

To evaluate a potential binary split, we calculate the weighted average of the Gini Impurities for the resulting subsets:

$$ G_{split} = \frac{n_{left}}{n} G_{left} + \frac{n_{right}}{n} G_{right} $$

Where:
- $n_{left}$ and $n_{right}$ are the number of instances in the left and right subsets
- $n$ is the total number of instances
- $G_{left}$ and $G_{right}$ are the Gini Impurities of the left and right subsets

The split with the lowest $G_{split}$ is chosen as the best split for that node.

## Mean Squared Error
For regression tasks, Mean Squared Error (MSE) is commonly used as the splitting criterion. MSE measures the average squared difference between the predicted and actual values.

The Mean Squared Error for a set of data points is calculated as:

$$ MSE(set) = \frac{1}{|set|} \sum_{i \in set} (y_i - \hat{y})^2 $$

Where:
- $|set|$ is the number of data points in the set
- $y_i$ is the actual value of the i-th data point
- $\hat{y}$ is the mean of the target values in the set, calculated as:

$$ \hat{y} = \frac{1}{|set|} \sum_{i \in set} y_i $$

To evaluate a potential binary split, we calculate the weighted average of the MSEs for the resulting subsets:

$$ MSE_{split} = \frac{n_{left}}{n} MSE_{left} + \frac{n_{right}}{n} MSE_{right} $$

Where:
- $n_{left}$ and $n_{right}$ are the number of instances in the left and right subsets
- $n$ is the total number of instances
- $MSE_{left}$ and $MSE_{right}$ are the Mean Squared Errors of the left and right subsets

The split with the lowest $MSE_{split}$ is chosen as the best split for that node in the regression tree.

## Pruning a Decision Tree

Pruning is the process of removing branches from a decision tree to prevent overfitting. Overfitting occurs when the tree is too complex and fits the training data too closely, capturing noise and details that are specific to the training data rather than generalizing to new, unseen data.

Pruning helps to simplify the tree, making it more robust and reducing its complexity. This can lead to better generalization to new data.

There are two main types of pruning:
1. Pre-pruning (Early Stopping):
   - This method stops the growth of the tree before it fully fits the training data.
   - It uses stopping criteria such as:
     - Maximum depth of the tree
     - Minimum number of samples required to split an internal node
     - Minimum number of samples required to be at a leaf node
   - Pre-pruning is computationally efficient but may result in underfitting if the stopping criteria are too strict.

2. Post-pruning (Reduced Error Pruning):
   - This method first grows a full tree and then removes branches that do not provide significant predictive power.
   - The process typically involves:
     1. Grow a full tree on the training data
     2. For each node:
        - Calculate the accuracy of the tree with and without the node
        - If removing the node increases accuracy, prune it
   - Post-pruning often results in better performance but is more computationally expensive than pre-pruning.





