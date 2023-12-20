import os

import dill
import keras
import matplotlib.pyplot as plt
from keras import layers, losses


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


load_data("data/vonDataset20180426.dill")


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
                # input_shape=(None, 150),
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
            layers.Dense(1, activation="relu", input_shape=(None, 150)),
            layers.Dropout(0.5),
            layers.Dense(1, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="relu"),
        ]
    )

    # hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        loss=losses.BinaryCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=[
            keras.metrics.BinaryAccuracy(threshold=0.5),
            keras.metrics.Precision(thresholds=0.0),
            keras.metrics.Recall(thresholds=0.0),
        ],
    )
    return model


# Step 3: Train and test the model
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# Main script
if __name__ == "__main__":
    # Step 1: Load data
    X_train, y_train, X_test, y_test, X_val, y_val = load_data(
        "data/vonDataset20180426.dill"
    )

    # # Step 2: Define the model hyperparameters
    # load_model = False
    # if load_model:
    #     model = keras.models.load_model("models/dnn.keras")
    # else:
    #     tuner = kt.Hyperband(
    #         model_builder_cnn_character_level,
    #         objective="val_binary_accuracy",
    #         max_epochs=10,
    #         factor=3,
    #         directory="my_dir",
    #         project_name="intro_to_kt",
    #     )
    #
    #     stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    #
    #     tuner.search(
    #         X_train, validation_data=[X_val], epochs=50, callbacks=[stop_early]
    #     )
    #
    #     # Get the optimal hyperparameters
    #     best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    #     print(best_hps)
    #
    #     model = tuner.hypermodel.build(best_hps)
    model = model_builder_cnn_character_level()
    epochs = 10
    history = model.fit(
        X_train,
        y_train,
        validation_data=[X_val],
        epochs=epochs,
    )

    if not os.path.exists("models"):
        os.mkdir("models")

    model.save("models/dnn.keras")

    loss, accuracy = model.evaluate(X_test)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    history_dict = history.history
    acc = history_dict["binary_accuracy"]
    val_acc = history_dict["val_binary_accuracy"]
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
