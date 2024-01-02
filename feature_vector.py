import re
from urllib.parse import urlparse

import numpy as np
import tldextract
import unicodedata
from tqdm import tqdm


# transformation of the data into a feature vector
# using https://www.tensorflow.org/text/tutorials/word2vec, https://pypi.org/project/tldextract/

# other data to vector transformation possibility:
# using handcrafted features
# /[4]%20PHISH-SAFE%20URL%20Features%20based%20Phishing%20Detection%20System%20using%20Machine%20Learning.pdf , [5] LexicalFeatureSelection.pdf  , [6] EfficientDeepLearningPhishingDetection.pdf


# using word detection and random word detection (really complex approach)
# /[3] NLPBasedPhishingAttack.pdf , /[2] ESWA_Sahingoz_Makale.pdf , [1] Phishing_URL_Detection_A_Real-Case_Scenario_Through_Login_URLs.pdf


def feature_vector(x: np.ndarray[str]) -> np.ndarray[int]:
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
    vector = np.zeros([x.shape[0], 23])
    # handcrafted straight forward features
    for i, url in enumerate(tqdm(x)):
        url_tld = tldextract.extract(url)  # https://pypi.org/project/tldextract/
        url_parsed = urlparse(
            "//www.cwi.nl:80/%7Eguido/Python.html"
        )  # https://docs.python.org/3/library/urllib.parse.html
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
        vector[i][j] = count_suspicious_words(url)
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
    # word detection and random word detection
    ### TODO  add more high level features
    print(j)
    return vector


def count_suspicious_chars(url):
    suspicious_chars = set("%#^$*&!’,:")
    suspicious_chars_count = sum(1 for char in url if char in suspicious_chars)
    return suspicious_chars_count


def count_suspicious_words(url):
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
