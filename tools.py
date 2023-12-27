import tldextract
from tqdm import tqdm
import numpy
from sklearn import preprocessing, svm, tree, ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


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


def feature_vector(x: numpy.ndarray[str]) -> numpy.ndarray[int]:
    """
    functions that transform the data into a feature vector, inspired by the article PHISH-SAFE URL Features based Phishing Detection System using Machine Learning.pdf
    feature 1 : number of dots in the URL
    feature 2 : number of hyphens in the URL
    feature 3 : number of @ in the URL
    feature 4 : length of the URL
    feature 5 : number of digits in the URL
    feature 6 : number of // in the URL
    feature 7 : use of "HTTPS"
    feature 8 : number of time "HTTP" appears
    feature 9 : 1 if the IP address is used rather than the domain, 0 otherwise
    feature 10 : detection of suspicious word. equals to the number of the suspicious words, the suspicious word being in the list [token, confirm, security, PayPal, login, signin, bank, account, update]
    feature 11 : Position of Top-Level Domain: This feature checks the position of top-level domain at proper place in URL.

    Have been used in the article but won't be used in our implementation:
    Domains count in URL: Phishing URL may contain more than one domain in URL. Two or more domains is used to redirect address.
    DNS lookup: If the DNS record is not available then the website is phishing. The life of phishing site is very short, therefore; this DNS information may notbe available after some time.
    Inconsistent URL: If the domain name of suspicious web page is not matched  with the WHOIS database record, then the web page is considered as phishing.
    Age of Domain: If the age of website is less than 6 month, then chances of fake web page are more.

    :param x:
    :return a vector of size number of URL * 11 :
    """
    vector = numpy.zeros([x.shape[0], 11])
    # handcrafted straight forward features
    for i, url in enumerate(tqdm(x)):
        url_tld = tldextract.extract(url)  # https://pypi.org/project/tldextract/
        vector[i][0] = url.count(".")
        vector[i][1] = url.count("-")
        vector[i][2] = url.count("@")
        vector[i][3] = len(url)
        vector[i][4] = sum(c.isdigit() for c in url)
        vector[i][5] = url.count("//")
        vector[i][6] = url.count("https")
        vector[i][7] = url.count("http")
        vector[i][8] = int(url_tld.domain == url_tld.ipv4)
        # suspicious word detection
        suspicious_words = [
            "token",
            "confirm",
            "security",
            "PayPal",
            "login",
            "signin",
            "bank",
            "account",
            "update",
        ]
        for word in suspicious_words:
            vector[i][9] += url.count(word)
        # top level domain detection: give the rank of the first character of the top level domain in the url
        vector[i][10] = url.find(url_tld.suffix)

    # word detection and random word detection
    ### TODO  add more high level features

    return vector


## TODO feature ranking (Information Gain?)


def machine_learning_models(x: numpy.ndarray[int], y: numpy.ndarray[int]):
    # Preprocessing for logistic regression and svm
    scaler = preprocessing.StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    # Implementation of Logistic Regression
    logistic_classifier = LogisticRegression(penalty="l2")
    logistic_classifier.fit(x_scaled, y)
    logistic_score = logistic_classifier.score(x_scaled, y)
    print("logistic_score", logistic_score)
    # implementation of Naive Bayes
    bayes_classifier = MultinomialNB(force_alpha=True)
    bayes_classifier.fit(x, y)
    bayes_score = bayes_classifier.score(x, y)
    print("bayes_score", bayes_score)
    # Implementation on support vector machines
    svm_classifier = svm.SVC(gamma="auto", max_iter=100)
    svm_classifier.fit(x_scaled, y)
    svm_score = svm_classifier.score(x_scaled, y)
    print("svm_score", svm_score)
    return logistic_classifier, bayes_classifier, svm_classifier


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def transform_url_data():
    BATCH_SIZE = 1000

    raw_train_ds = tf.data.experimental.make_csv_dataset(
        "data/train_x.csv",
        batch_size=BATCH_SIZE,
        label_name="train_y",
        select_columns=["train_x", "train_y"],
        num_epochs=1,
    ).map(lambda x, y: (x["train_x"], y))

    raw_val_ds = tf.data.experimental.make_csv_dataset(
        "data/val_x.csv",
        batch_size=BATCH_SIZE,
        label_name="val_y",
        select_columns=["val_x", "val_y"],
        num_epochs=1,
    ).map(lambda x, y: (x["val_x"], y))

    raw_test_ds = tf.data.experimental.make_csv_dataset(
        "data/test_x.csv",
        batch_size=BATCH_SIZE,
        label_name="test_y",
        select_columns=["test_x", "test_y"],
        num_epochs=1,
    ).map(lambda x, y: (x["test_x"], y))

    MAX_FEATURES = 1000
    SEQUENCE_LENGTH = 150

    vectorize_layer = layers.TextVectorization(
        max_tokens=MAX_FEATURES,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
    )

    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    return train_ds, val_ds, test_ds
