import keras
from keras import layers


def model_builder_cnn(number_of_features=11):
    """
    Based on the model proposed by Efficient deep learning techniques for the detection of phishing websites
    Model relying on feature vectors extracted from the URL
    :param hp:
    :return:
    """

    # Build the model
    model = keras.Sequential(
        [
            layers.Reshape((number_of_features, 1), input_shape=(number_of_features,)),
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
