import keras
from keras import layers

#    "DNN with 5 hidden layers observe the best results"
def model_builder_dnn(number_of_features=11):
    """
    Based on the model proposed by Efficient deep learning techniques for the detection of phishing websites
    Model relying on feature vectors extracted from the URL
    :param hp:
    :return:
    """

    # Build the model
    model = keras.Sequential(
        [
            layers.InputLayer(input_shape=(number_of_features,)),
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
