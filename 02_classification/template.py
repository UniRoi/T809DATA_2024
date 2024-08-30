# Author: Matthias Reiser
# Date: 28.8.24
# Project: Computer Exercise 2
# Acknowledgements: gpt4
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, multivariate_normal


def gen_data(
    n: int,
    locs: np.ndarray,
    scales: np.ndarray
) -> np.ndarray:
    '''
    Return n data points, their classes and a unique list of all classes, from each normal distributions
    shifted and scaled by the values in locs and scales
    Mostly done by gpt4
    '''

    data = [] # create empty list
    classes = [] # create empty list
    class_list = list(range(len(locs)))
    
    for i, (loc, scale) in enumerate(zip(locs, scales)): # iterate through the lists and do .rvs with only one value for loc and scale
        point = norm.rvs(loc=loc, scale=scale, size=n)
        data.extend(point) # append adds a single point to a list
        classes.extend([i] * n) # extend adds multible variables to the list
    
    return np.array(data), np.array(classes), class_list

def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    selected_features = features[targets == selected_class]

    return np.mean(selected_features, axis=0)


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    selected_features = features[targets == selected_class]

    return np.cov(selected_features, rowvar = False)


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    # return norm.pdf(x=feature, loc=class_mean, scale=class_covar)
    return multivariate_normal.pdf(x=feature, mean=class_mean, cov=class_covar)
    


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array

    Is currently only supporting two classes
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))
    
    likelihoods = []
    for i in range(test_features.shape[0]):
        likely = []
        for class_label in classes:
            likely.append( likelihood_of_class(test_features[i], means[class_label], covs[class_label]))
        likelihoods.append(np.array(likely))
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    ...
    return np.argmax(likelihoods, axis=1)



if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    num_datapoints = 100
    mean1 = [-1, np.sqrt(5)]
    mean2 = [-4, np.sqrt(2)]
    cov1 = [1, np.sqrt(5)]
    cov2 = [1, np.sqrt(2)]
    
    def compare_arrays(prediction, targets):
        cnt = 0
        for i in range(len(prediction)):
            if prediction[i] == targets[i]:
                cnt += 1
        return (cnt/len(prediction)*100)
    # Section 1
    print("Section 1")
    # print(gen_data(1, [-1, 0, 1], [2, 2, 2]))
    features, targets, classes = gen_data(num_datapoints, mean1, cov1)
    # print("Data:", features)
    # print("datalen:", len(features))
    # print("Classes:", targets)
    # print("Classeslen:", len(targets))

    # print("Class List:", classes)
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.8)
    

    # Section 2
    # print("Section 2")
    # better would be if seperated into different arrays based on classes and then do the print 
    # num_classes = len(classes)
    # num_per_class = int(len(targets)/num_classes)

    # num_feat_per_class = np.zeros(num_per_class)
    # print(features[0:num_per_class])
    # print(len(features[0:num_per_class]))
    # print(len(features[num_per_class:]))
    # print(len(num_feat_per_class))

    # plt.scatter(features[0:num_per_class], num_feat_per_class, label=f'Class {targets[0]}')
    # plt.scatter(features[num_per_class:], num_feat_per_class, marker="x", color='orange', label=f'Class {targets[num_per_class+1]}')
    # plt.legend(loc='upper center')
    # plt.savefig("T809DATA_2024/02_classification/2_1.png")
    # plt.show()
    
    # Section 3
    # print("Section 3")
    # print(mean_of_class(train_features, train_targets, 0))
    # print(mean_of_class(train_features, train_targets, 1))

    # # Section 4
    # print("Section 4")
    # print(covar_of_class(train_features, train_targets, 0))
    # print(covar_of_class(train_features, train_targets, 1))

    # Section 5
    # print("Section 5")
    # class_mean = mean_of_class(train_features, train_targets, 0)
    # class_cov = covar_of_class(train_features, train_targets, 0)
    # print("Likelihood of feature ", likelihood_of_class(test_features[0:3], class_mean, class_cov))
    # print("Test targets ", test_targets[0:3])

    # Section 6
    print("Section 6")
    likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    # print(likelihoods)

    # Section 7
    print("Section 7")
    prediction = predict(likelihoods)
    print("prediction: ", prediction)
    print("Validation: ", test_targets)

    print("percentage: ", compare_arrays(prediction, test_targets))

    # Section 8
    print("Section 8")
    features, targets, classes = gen_data(num_datapoints, mean2, cov2)
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.8)
    likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    prediction = predict(likelihoods)
    print("prediction sec 8: ", prediction)
    print("Validation sec 8: ", test_targets)

    print("percentage sec 8: ", compare_arrays(prediction, test_targets))

    pass
