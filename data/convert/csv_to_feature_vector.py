import os
import re
from urllib.parse import urlparse

import numpy as np
import tldextract
import unicodedata
from fuzzywuzzy import process
from sklearn.feature_selection import SelectKBest, chi2
from tqdm import tqdm

from data.load.load import load_csv


# transformation of the data into a feature vector
# using https://www.tensorflow.org/text/tutorials/word2vec, https://pypi.org/project/tldextract/

# other data to vector transformation possibility:
# using handcrafted features
# /[4]%20PHISH-SAFE%20URL%20Features%20based%20Phishing%20Detection%20System%20using%20Machine%20Learning.pdf , [5] LexicalFeatureSelection.pdf  , [6] EfficientDeepLearningPhishingDetection.pdf


# using word detection and random word detection (really complex approach)
# /[3] NLPBasedPhishingAttack.pdf , /[2] ESWA_Sahingoz_Makale.pdf , [1] Phishing_URL_Detection_A_Real-Case_Scenario_Through_Login_URLs.pdf


def feature_vector(x):
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
    vector = np.zeros([x.shape[0], 26])

    url_parsed = urlparse(
        "//www.cwi.nl:80/%7Eguido/Python.html"
    )  # https://docs.python.org/3/library/urllib.parse.html

    # handcrafted straight forward features
    for i, url in enumerate(tqdm(x)):
        try:
            url_tld = tldextract.extract(url)  # https://pypi.org/project/tldextract/
        except:
            print(i, url)
            continue

        j = 0
        vector[i][j] = url.count(".")
        j += 1
        vector[i][j] = url.count("-")
        j += 1
        vector[i][j] = url.count("@")
        j += 1
        vector[i][j] = url.count("?")
        j += 1
        vector[i][j] = len(url)
        j += 1
        vector[i][j] = sum(c.isdigit() for c in url)
        j += 1
        vector[i][j] = url.count("/")
        j += 1
        vector[i][j] = url.count("//")
        j += 1
        vector[i][j] = url.count("https")
        j += 1
        vector[i][j] = url.count("http")
        j += 1
        vector[i][j] = url.count("www")
        j += 1
        vector[i][j] = int(url_tld.domain == url_tld.ipv4)
        j += 1
        # suspicious word detection
        vector[i][j] = count_suspicious_study_words(url)
        j += 1
        vector[i][j] = count_suspicious_gift_cards_words(url)
        j += 1
        # top level domain detection: give the rank of the first character of the top level domain in the url
        vector[i][j] = url.find(url_tld.suffix)
        j += 1
        vector[i][j] = len(url_parsed.path) / len(url)
        j += 1
        vector[i][j] = count_suspicious_chars(url)
        j += 1
        # presence of a symbol as the last character
        vector[i][j] = url[-1] in "%#^$*&!’,:"
        j += 1
        # number of redirection occurrence
        vector[i][j] = url_tld.subdomain.count(".")
        j += 1
        # presence of IP address
        vector[i][j] = check_ip_address_presence(url)
        j += 1
        # number of subdomains
        vector[i][j] = len(url_tld.subdomain.split("."))
        j += 1
        # presence of port number
        vector[i][j] = check_port_presence(url)
        j += 1
        # number of unicode characters
        vector[i][j] = sum(1 for char in url if unicodedata.category(char) != "Cc")
        j += 1
        # whether there is a query in the url
        vector[i][j] = int(bool(url_parsed.query))
        j += 1
        # Finds the ratio of total number of special characters
        # to the length of URL
        vector[i][j] = sum(1 for char in url if not char.isalnum()) / len(url)
        j += 1
        # check whether the URL contains misspelled words
        vector[i][j] = is_misspelled(url)
    # word detection and random word detection
    print(f"Number of features: {j}")
    return vector


def count_suspicious_chars(url):
    suspicious_chars = set("%#^$*&!’,:")
    suspicious_chars_count = sum(1 for char in url if char in suspicious_chars)
    return suspicious_chars_count


def count_suspicious_study_words(url):
    sum_suspicious_words = 0
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
        "submit",
        "logon",
        "suspend",
        "secure",
        "webscr",
        "cmd",
        "wp",
        "payment",
        "home",
        "dropbox",
        "webhostapp",
    ]
    for word in suspicious_words:
        sum_suspicious_words += url.count(word)
    return sum_suspicious_words


def count_suspicious_gift_cards_words(url):
    sum_suspicious_words = 0
    suspicious_words = ["giftcard", "gift_card", "gift-card"]
    for word in suspicious_words:
        sum_suspicious_words += url.count(word)
    return sum_suspicious_words


def check_ip_address_presence(url):
    # Regular expression to match an IPv4 address
    ip_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    match = ip_pattern.search(url)
    if match:
        return 1
    return 0


def check_port_presence(url):
    # Regular expression to match a port number in the URL
    port_pattern = re.compile(r":\d{1,5}")
    match = port_pattern.search(url)
    if match:
        return 1
    return 0


def is_misspelled(string, threshold_low=80, threshold_high=98):
    """
    Function checks whether there are at least one misspelling in a URL by comparing the spell of words used with the real
    spelling 25 regular misspelled words in phishing URL. Words are retrieved by splitting between any non letter characters

    :param string: URL
    :param threshold_low: minimum threshold of similarity to be considered matching one of the common legit names
    :param threshold_high: maximum threshold of similarity to be considered misspelled compared to the legit name
    :return: an approximation of the number of misspelled words that the URL contains. Do not take misspelling using
    numbers into account
    """
    legit_names_misspelled = [
        "google",
        "microsoft",
        "facebook",
        "apple",
        "amazon",
        "paypal",
        "linkedin",
        "yahoo",
        "twitter",
        "instagram",
        "netflix",
        "ebay",
        "spotify",
        "adobe",
        "dropbox",
        "wordpress",
        "github",
        "uber",
        "airbnb",
        "visa",
        "mastercard",
        "americanexpress",
        "walmart",
        "cisco",
        "oracle",
    ]
    words = re.split(r"[^a-zA-Z]", string)  # split the URL in parts of words
    words = [word for word in words if word]  # remove empty words
    for word in words:
        # Check if the word is misspelled by comparing it to the list of common legit names
        match, score = process.extractOne(word, legit_names_misspelled)
        if threshold_low <= score <= threshold_high:
            return 1

    return 0


def main():
    train_x, train_y, val_x, val_y, test_x, test_y = load_csv()

    train_x = feature_vector(train_x)
    val_x = feature_vector(val_x)
    test_x = feature_vector(test_x)

    p = os.path.dirname(os.path.abspath(__file__)) + "/.."

    os.makedirs(f"{p}/feature_vector", exist_ok=True)

    np.savez_compressed(f"{p}/feature_vector/train_y.npz", train_y.to_numpy())
    np.savez_compressed(f"{p}/feature_vector/val_y.npz", val_y.to_numpy())
    np.savez_compressed(f"{p}/feature_vector/test_y.npz", test_y.to_numpy())

    # for the top k features
    for k in [10, 20, 30, 40]:
        os.makedirs(f"feature_vector/{k}", exist_ok=True)

        # # INFORMATION GAIN feature selection - better suited for correlated data
        #
        # # Calculate Information Gain using mutual_info_classif
        # info_gain_selector = SelectKBest(mutual_info_classif, k=10)
        # info_gain_selector.fit(train_x, train_df["train_y"])
        #
        # # Get the information gain scores and corresponding feature names
        # info_gain_scores = info_gain_selector.scores_
        #
        # # Print information gain scores and feature names
        # print("Information Gain Scores:")
        # for feature, score in enumerate(info_gain_scores):
        #     print(f"{feature}: {score}")

        # CHI SQUARE

        chi2_selector = SelectKBest(score_func=chi2, k="all")
        chi2_selector.fit(train_x, train_y)

        # Get the chi-square scores and corresponding feature names
        chi2_scores = chi2_selector.scores_

        # Print chi-square scores and feature names
        print("\nChi-Square Scores:")
        for feature, score in enumerate(chi2_scores):
            print(f"{feature}: {score}")

        # convert the data into a feature vector with only the selected features
        selected_feature_indices = chi2_selector.get_support(indices=True)
        train_x_k = train_x[:, :selected_feature_indices]
        val_x_k = val_x[:, :selected_feature_indices]
        test_x_k = test_x[:, :selected_feature_indices]

        np.savez_compressed(f"{p}/feature_vector/{k}/train_x.npz", train_x_k)
        np.savez_compressed(f"{p}/feature_vector/{k}/val_x.npz", val_x_k)
        np.savez_compressed(f"{p}/feature_vector/{k}/test_x.npz", test_x_k)


if __name__ == "__main__":
    main()
