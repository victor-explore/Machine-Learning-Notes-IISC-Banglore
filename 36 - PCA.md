# Principal Component Analysis (PCA)

## Problem Setup

Data: Let the dataset be $X = \{x_i\}_{i=1}^N$, where each data point $x_i \in \mathbb{R}^d$.

Goal: We want to reduce the dimensionality of the data from $d$ to $k$ while preserving as much of the data's variance as possible.

## Intuition

PCA aims to find a new coordinate system for the data that maximizes the variance along the first coordinate (principal component), while minimizing the variance along the other coordinates. This is useful for data visualization, noise reduction, and feature extraction.

## Steps

1. Standardize the data: Subtract the mean of each feature from the data.
 $$
 \mu_j = \frac{1}{N} \sum_{i=1}^N x_{ij}
 $$
 $$
 \hat{x}_i = x_i - \mu_j
 $$

2. Compute the covariance matrix:
 $$
 \Sigma = \frac{1}{N-1} \sum_{i=1}^N \hat{x}_i \hat{x}_i^T
 $$

3. Compute the eigenvectors and eigenvalues of the covariance matrix.

Eigenvectors represent the directions(principal components) that have maximum variance whereas the eigenvalues represent the variance along these directions.


4. Sort the eigenvectors by the eigenvalues in descending order.

5. Choose the top $k$ eigenvectors ie eigenvectors with the largest eigenvalues.

6. Project the data onto the eigenvectors to get the principal components.

