import dill
import keras
import pandas as pd
from keras import layers, losses
from keras.utils import to_categorical
from matplotlib import pyplot as plt

import tensorflow as tf

# This line must be executed before any other TensorFlow-related code
gpus = tf.config.experimental.list_physical_devices('GPU')
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
        metrics=[keras.metrics.Accuracy(), keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.F1Score()]
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

    checkpoint_path_loss = "cp.loss.model.keras"
    checkpoint_path_accuracy = "cp.accuracy.model.keras"

    try:
        model = keras.models.load_model(checkpoint_path_loss)
        print("Model loaded successfully!")
    except Exception as e:
        print("No model found or error in loading. Building a new model.")
        print("Error:", e)
        model: keras.Model = model_builder_cnn_character_level()

    model.summary()

    log_path = "logs.csv"

    try:
        history_df = pd.read_csv(log_path)
        print("Model history loaded successfully!")
    except Exception as e:
        print("No model history found or error in loading. Building a new history.")
        print("Error:", e)
        # create empty df
        history_df = pd.DataFrame()

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        callbacks=[
            keras.callbacks.ModelCheckpoint(checkpoint_path_loss, save_best_only=True, monitor='val_loss', mode='min'),
            keras.callbacks.ModelCheckpoint(checkpoint_path_accuracy, save_best_only=True, monitor='val_accuracy',
                                            mode='max'),
            keras.callbacks.CSVLogger(log_path, append=True),
        ],
        initial_epoch=history_df.shape[0],
        epochs=30,
    )

    # reload history after training
    history_df = pd.read_csv(log_path)

    metrics = ['accuracy', 'loss', 'precision', 'recall']
    for metric in metrics:
        plt.plot(history_df[metric])
        plt.plot(history_df['val_' + metric])
        plt.title('Model ' + metric)
        plt.ylabel(metric)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()
        plt.close()
