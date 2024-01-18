import keras
from keras import layers, losses


def model_builder_lstm(number_of_features=11):
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

    # Build the model
    model = keras.Sequential(
        [
            # input shape to change because of the change of interpretation of the features
            layers.Reshape((number_of_features, 1), input_shape=(number_of_features,)),
            layers.LSTM(units=number_of_features, return_sequences=True),
            layers.LSTM(units=number_of_features, return_sequences=True),
            layers.LSTM(units=number_of_features, return_sequences=True),
            layers.LSTM(units=number_of_features, return_sequences=False),
            layers.Dense(units=2, activation="sigmoid"),
        ]
    )

    hp_initial_learning_rate = 0.01

    model.compile(
        loss=losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=hp_initial_learning_rate),
        metrics=[
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.F1Score(),
        ]
    )

    return model
