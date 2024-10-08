# K Nearest neighbour (KNN)

## Basic idea

Recall that we had derived the following equation for non-parametric density estimation:

$$p(x) = \frac{k}{nV}$$

where:
- $k$ is the number of data points in region $R$
- $n$ is the total number of data points in the dataset $D$
- $V$ is the volume of the region $R$

In K-Nearest Neighbour (KNN) method, we fix the volume $V$ and count $k$

## Problem setting

Given a set of data points $D = \{x_1, y_1\}, \{x_2, y_2\}, ..., \{x_n, y_n\}$, where $x_i \in \mathbb{R}^d$ is the feature vector and $y_i \in \{1, 2, ..., C\}$ is the class label.

## Formulation

The posterior probability of class $c$ given input $x_i$ can be estimated as:

$$p(y=c|x_i) = \frac{p(x_i,y=c)}{p(x_i,y=1) + p(x_i,y=2) + ... + p(x_i  ,y=C)}$$

$$p(y=c|x_i) = \frac{\frac{k_i}{nV}}{\sum_{j=1}^C \frac{k_j}{nV}} = \frac{k_i}{\sum_{j=1}^C k_j}$$

Now we can use bayes classification rule to assign label to the new data point $x_i$.

## Algorithm

1. Choose a value for $k$
2. For each test data point $x_i$, find the $k$ nearest neighbours in the training data $D$
3. Assign the class label to $x_i$ based on the majority class label of the $k$ nearest neighbours
