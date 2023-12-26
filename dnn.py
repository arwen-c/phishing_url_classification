import keras
import pandas as pd
from keras import layers, losses
from matplotlib import pyplot as plt

from tools import feature_vector
from cnn_on_dill import model_builder_cnn_character_level

load_model = False

# TODO: determine the best values for the hyperparameters, like the learning rate

number_of_features: int = 11


def model_builder(hp):
    embedding_dim = hp.Int("embedding_dims", min_value=32, max_value=256, step=32)

    model = keras.Sequential(
        [
            layers.Embedding(10000, embedding_dim),  # Embedding layer
            layers.Dropout(0.25),  # Dropout layer to prevent overfitting
            layers.Convolution1D(kernel_size=5, filters=256),  # Convolutional layer
            layers.ELU(),  # ELU activation function
            layers.MaxPooling1D(pool_size=4),
            layers.Dropout(0.5),  # Dropout layer to prevent overfitting
            layers.LSTM(32),  # LSTM layer
            layers.Dropout(0.5),  # Dropout layer to prevent overfitting
            layers.Dense(1),  # Final Dense layer
        ]
    )

    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        loss=losses.BinaryCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        metrics=[
            keras.metrics.BinaryAccuracy(threshold=0.5),
            keras.metrics.Precision(thresholds=0.0),
            keras.metrics.Recall(thresholds=0.0),
        ],
    )

    return model


#    "DNN with 5 hidden layers observe the best results"
def model_builder_dnn(hp):
    """
    Based on the model proposed by Efficient deep learning techniques for the detection of phishing websites
    Model relying on feature vectors extracted from the URL
    :param hp:
    :return:
    """

    # Build the model
    model = keras.Sequential(
        [
            layers.InputLayer(input_shape=(11,)),  # shape = number of features
            layers.Dense(units=256, activation="relu"),
            layers.Dense(units=128, activation="relu"),
            layers.Dense(units=64, activation="relu"),
            layers.Dense(units=32, activation="relu"),
            layers.Dense(units=1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def model_builder_lstm(hp):
    """
    Based on the model proposed by Efficient deep learning techniques for the detection of phishing websites
    Model relying on feature vectors extracted from the URL
    :param hp:
    :return:
    Notes from the author on the paper
    "We modified the dataset dimensionality to implement LSTM.
    We have converted 10 features to 10 time-steps; each timestep
    consists of 1 feature. Hence, our dataset new dimension
    is (3526,10,1). Through LSTM, we attempted to find
    out the possible relationship between different features
    """
    model = keras.Sequential(
        [
            layers.Reshape((number_of_features, 1), input_shape=(number_of_features,)),
            layers.LSTM(
                units=number_of_features, return_sequences=True
            ),  # input shape to change because of the change of interpretation of the features
            layers.LSTM(units=number_of_features, return_sequences=True),
            layers.LSTM(units=number_of_features, return_sequences=True),
            layers.LSTM(units=number_of_features, return_sequences=False),
            layers.Dense(units=1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def model_builder_cnn(hp):
    """
    Based on the model proposed by Efficient deep learning techniques for the detection of phishing websites
    Model relying on feature vectors extracted from the URL
    :param hp:
    :return:
    """
    model = keras.Sequential(
        [
            layers.Reshape(
                (number_of_features, 1),
                input_shape=(number_of_features,),
            ),
            layers.Conv1D(filters=32, kernel_size=3, activation="tanh"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=16, kernel_size=3, activation="tanh"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=128, kernel_size=3, activation="tanh"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=256, kernel_size=3, activation="tanh"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=512, kernel_size=3, activation="tanh"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=1024, kernel_size=3, activation="tanh"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(units=500, activation="tanh"),
            layers.Dense(units=1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = model_builder_cnn(None)
    model.summary()

    # import the csv data from train_x.csv, val_x.csv and test_x.csv
    train_df = pd.read_csv("data/train_x.csv")
    val_df = pd.read_csv("data/val_x.csv")
    test_df = pd.read_csv("data/test_x.csv")

    # transform the data into a feature vector
    train_x = feature_vector(train_df["train_x"])
    val_x = feature_vector(val_df["val_x"])
    test_x = feature_vector(test_df["test_x"])

    if load_model:
        model = keras.models.load_model("models/dnn.keras")
    else:
        epochs = 10
        history = model.fit(
            train_x,
            train_df["train_y"],
            validation_data=[val_x],
            epochs=epochs,
        )

        # if not os.path.exists("models"):
        #     os.mkdir("models")
        #
        # model.save("models/dnn.keras")

    loss, accuracy = model.evaluate(test_x)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    history_dict = history.history
    print(history_dict.keys())
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, "bo", label="Training loss")
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    plt.show()
