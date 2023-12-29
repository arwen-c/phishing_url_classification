import numpy
import tldextract
from tqdm import tqdm


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