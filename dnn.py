import os

import keras
import keras_tuner as kt
import tensorflow as tf
from keras import layers, losses
from matplotlib import pyplot as plt

load_model = False


def model_builder(hp):
    embedding_dim = hp.Int('embedding_dims', min_value=32, max_value=256, step=32)

    model = keras.Sequential([
        layers.Embedding(MAX_FEATURES, embedding_dim),  # Embedding layer
        layers.Dropout(0.25),  # Dropout layer to prevent overfitting
        layers.Convolution1D(kernel_size=5, filters=256),  # Convolutional layer
        layers.ELU(),  # ELU activation function
        layers.MaxPooling1D(pool_size=4),
        layers.Dropout(0.5),  # Dropout layer to prevent overfitting
        layers.LSTM(32),  # LSTM layer
        layers.Dropout(0.5),  # Dropout layer to prevent overfitting
        layers.Dense(1)  # Final Dense layer
    ])

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=[
                      keras.metrics.BinaryAccuracy(threshold=0.5),
                      keras.metrics.Precision(thresholds=0.0),
                      keras.metrics.Recall(thresholds=0.0),
                  ])

    return model


if __name__ == "__main__":
    BATCH_SIZE = 1000

    raw_train_ds = tf.data.experimental.make_csv_dataset(
        "data/train_x.csv",
        batch_size=BATCH_SIZE,
        label_name='train_y',
        select_columns=["train_x", "train_y"],
        num_epochs=1,
    ).map(lambda x, y: (x['train_x'], y))

    raw_val_ds = tf.data.experimental.make_csv_dataset(
        "data/val_x.csv",
        batch_size=BATCH_SIZE,
        label_name='val_y',
        select_columns=["val_x", "val_y"],
        num_epochs=1,
    ).map(lambda x, y: (x['val_x'], y))

    raw_test_ds = tf.data.experimental.make_csv_dataset(
        "data/test_x.csv",
        batch_size=BATCH_SIZE,
        label_name='test_y',
        select_columns=["test_x", "test_y"],
        num_epochs=1,
    ).map(lambda x, y: (x['test_x'], y))

    MAX_FEATURES = 1000
    SEQUENCE_LENGTH = 150

    vectorize_layer = layers.TextVectorization(
        max_tokens=MAX_FEATURES,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH,
    )

    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)


    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label


    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    if load_model:
        model = keras.models.load_model('models/dnn.keras')
    else:
        tuner = kt.Hyperband(model_builder,
                             objective='val_binary_accuracy',
                             max_epochs=10,
                             factor=3,
                             directory='my_dir',
                             project_name='intro_to_kt')

        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        tuner.search(train_ds, validation_data=val_ds, epochs=50, callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hps)

        model = tuner.hypermodel.build(best_hps)

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    if not os.path.exists("models"):
        os.mkdir('models')

    model.save('models/dnn.keras')

    loss, accuracy = model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    history_dict = history.history
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()
