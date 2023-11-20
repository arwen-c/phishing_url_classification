import numpy
from sklearn import svm, gaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def data_cleansing(dataframe):
    # Remove duplicates and update the dataframe
    dataframe = dataframe.drop_duplicates()

    # remove the data with missing values
    dataframe = dataframe.dropna(subset=["URL"])
    pass


# transformation of the data into a feature vector
# using https://www.tensorflow.org/text/tutorials/word2vec, https://pypi.org/project/tldextract/

# other data to vector transformation possibility:
# using handcrafted features
# /[4]%20PHISH-SAFE%20URL%20Features%20based%20Phishing%20Detection%20System%20using%20Machine%20Learning.pdf , [5] LexicalFeatureSelection.pdf  , [6] EfficientDeepLearningPhishingDetection.pdf


# using word detection and random word detection (really complex approach)
# /[3] NLPBasedPhishingAttack.pdf , /[2] ESWA_Sahingoz_Makale.pdf , [1] Phishing_URL_Detection_A_Real-Case_Scenario_Through_Login_URLs.pdf


def feature_vector(x):
    # handcrafted features

    # word detection and random word detection
    pass


def machine_learning_models (x :numpy.ndarray[int],y :numpy.ndarray[int]):
    # Implementation of Logistic Regression
    logistic_classifier = LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto',verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    logistic_classifier.fit(x, y)
    logistic_score = logistic_classifier.score(x, y)
    print("logistic_score", logistic_score)
    # Implementation on support vector machines
    svm_classifier = svm.SVC()
    svm_classifier.fit(x, y)
    svm_score = svm_classifier.score(x, y)
    print("svm_score", svm_score)
    #implementation of Naive Bayes
    bayes_classifier = MultinomialNB(force_alpha=True)
    bayes_classifier.fit(x, y)
    bayes_score = bayes_classifier.score(x, y)
    print("bayes_score",bayes_score)
    return logistic_score,svm_score,bayes_score


