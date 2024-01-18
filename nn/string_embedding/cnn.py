import keras
from keras import layers, losses


def model_builder_cnn_character_level(number_of_features=26):
    """
    Based on the model proposed by An Effective Phishing Detection Model Based on Character Level Convolutional Neural Network from URL
    Model relying on character level embedding as an input
    :param character_embedding: character level embedding
    :param hp: hyperparameters
    :return:
    """

    # Build the model
    model = keras.Sequential(
        [
            layers.Input((150, 1)),
            layers.Convolution1D(
                kernel_size=3,
                filters=256,
                activation="relu",
            ),
            layers.MaxPooling1D(pool_size=3),
            layers.Convolution1D(kernel_size=3, filters=256, activation="relu"),
            layers.Convolution1D(kernel_size=3, filters=256, activation="relu"),
            layers.Convolution1D(kernel_size=3, filters=256, activation="relu"),
            layers.Convolution1D(kernel_size=3, filters=256, activation="relu"),
            layers.Convolution1D(kernel_size=7, filters=256, activation="relu"),
            layers.Convolution1D(kernel_size=7, filters=256, activation="relu"),
            layers.MaxPooling1D(pool_size=3),
            layers.Flatten(),
            layers.Dense(2048, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(2048, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ]
    )

    hp_initial_learning_rate = 0.0001

    model.compile(
        loss=losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=hp_initial_learning_rate),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.F1Score(),
        ]
    )

    return model
