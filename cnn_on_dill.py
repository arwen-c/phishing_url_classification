import dill
import keras
import keras_tuner
from keras import layers, losses
from tensorflow.keras.utils import to_categorical

import tensorflow as tf

# This line must be executed before any other TensorFlow-related code
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# 1: Load data from .dill file
def load_data(dill_file):
    with open(dill_file, "rb") as f:
        pickleData = dill.load(f)
        ds_train_raw_x, ds_train_raw_y = pickleData["train_x"], pickleData["train_y"]
        ds_val_raw_x, ds_val_raw_y = pickleData["val_x"], pickleData["val_y"]
        ds_test_raw_x, ds_test_raw_y = pickleData["test_x"], pickleData["test_y"]
        print(ds_train_raw_y)
        print(ds_test_raw_x.shape)
        return (
            ds_train_raw_x,
            ds_train_raw_y,
            ds_test_raw_x,
            ds_test_raw_y,
            ds_val_raw_x,
            ds_val_raw_y,
        )


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
            layers.Reshape((150, 1), input_shape=(150,)),
            layers.Convolution1D(
                kernel_size=3,
                filters=256,
                activation="relu",
            ),
            layers.MaxPooling1D(pool_size=3),
            layers.Convolution1D(
                kernel_size=3, filters=256, activation="relu"
            ),  # use of relu because it learns faster than sigmoid
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
            layers.Dense(2, activation="softmax"),
        ]
    )

    # hp_initial_learning_rate = hp.Float('initial_learning_rate', min_value=0.0001, max_value=0.001, sampling='log')
    hp_initial_learning_rate = 0.001
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
            keras.metrics.Accuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.F1Score(),
        ],
    )

    return model


if __name__ == "__main__":
    # Step 1: Load data
    X_train, y_train, X_test, y_test, X_val, y_val = load_data(
        "data/vonDataset20180426.dill"
    )

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)

    model = model_builder_cnn_character_level()

    model.summary()

    history = model.fit(
        X_train,
        y_train,
        batch_size=256,
        validation_data=[X_val],
        epochs=10,
    )

    tuner = keras_tuner.GridSearch(
        model_builder_cnn_character_level,
        objective="val_accuracy",
        max_epochs=100,
        directory="tuner_cp",
        project_name="cnn_character_level",
    )
    #
    # stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    #
    # tuner.search(X_train, y_train, epochs=50, validation_data=[X_val], callbacks=[stop_early])
    #
    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    #
    # print(f"""
    # The hyperparameter search is complete. The optimal number of units in the first densely-connected
    # layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    # is {best_hps.get('learning_rate')}.
    # """)
    #
    # # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    # model = tuner.hypermodel.build(best_hps)
    # history = model.fit(X_train, y_train, epochs=100, validation_data=[X_val])
    #
    # val_acc_per_epoch = history.history['val_accuracy']
    # best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    # print('Best epoch: %d' % (best_epoch,))
    #
    # hypermodel = tuner.hypermodel.build(best_hps)
    #
    # # Retrain the model
    # hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_data=[X_val])
    #
    # eval_result = hypermodel.evaluate(X_test, y_test)
    # print("[test loss, test accuracy]:", eval_result)
