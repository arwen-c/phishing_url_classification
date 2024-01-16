import keras
from keras import layers, losses


# 2: Define the CNN model using Keras
def model_builder_cnn_character_level(hp=None):
    """
    Based on the model proposed by An Effective Phishing Detection Model Based on Character Level Convolutional Neural Network from URL
    Model relying on character level embedding as an input
    :param character_embedding: character level embedding
    :param hp: hyperparameters
    :return:
    """
    # character level embedding given as input
    # CONVOLUTIONAL NEURAL NETWORK
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

    # hp_initial_learning_rate = hp.Float('initial_learning_rate', min_value=0.0001, max_value=0.001, sampling='log')
    hp_initial_learning_rate = 0.0001
    # hp_decay_steps = hp.Int('decay_steps', min_value=100, max_value=1000, step=100)
    # hp_decay_rate = hp.Float('decay_rate', min_value=0.9, max_value=0.99, sampling='log')

    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=hp_initial_learning_rate,
    #     decay_steps=hp_decay_steps,
    #     decay_rate=hp_decay_rate,
    #     staircase=True)

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
