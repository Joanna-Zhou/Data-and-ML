'''
Implement the k-NN algorithm for regression with three different distance metrics (l2, l1, and l∞). Use 5-fold cross-validation1 to estimate k, and the preferred distance metric using a root-mean-square error (RMSE) loss. Compute nearest neighbours using a brute-force approach.
Apply your algorithm to all regression datasets (use n train=1000, d=2 for rosenbrock). For each dataset, report the estimated value of k and the preferred distance metric, and report the cross-validation RMSE and test RMSE with these settings. Format these results in a table.
Plot the cross-validation prediction curves (merging the predictions from all splits) for the one-dimensional regression dataset mauna loa at several values of k for the l2 distance metric. In separate figures, plot the prediction on the test set, as well as the cross-validation loss across k for this model. Discuss your results.
'''
# import pandas
from data_utils import load_dataset
