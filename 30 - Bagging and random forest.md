# Bagging
   - Let the original data be D.
   - Create n new datasets by sampling with replacement from D.
   - Each new dataset will have the same number of samples as the original dataset, but some samples will be repeated, and some will be excluded.
   - Train a model on each of the new datasets.
   - The final prediction is made by aggregating the predictions of all models:
     - For classification, the final prediction is made by taking the majority vote of the predictions of all models.
     - For regression, the final prediction is made by taking the average of the predictions of all models.

# Random Forest - Bagged descion trees
   - Bagging of decision trees.
   - Each decision tree is trained on a random subset of the features.
   - The final prediction is made by taking the majority vote of the predictions of all decision trees.

