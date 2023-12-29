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
