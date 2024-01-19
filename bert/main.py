import gc

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import TFBertForSequenceClassification, BertTokenizer
from keras.callbacks import CSVLogger
from matplotlib import pyplot as plt

from data.load.load import load_csv

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4000)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')


def encode(urls):
    input_ids, attention_masks = [], []

    for url in tqdm(urls):
        inputs = tokenizer.encode_plus(
            url,
            max_length=160,
            padding="max_length",
            return_tensors='tf',
        )
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])

    print("Encoding complete!")
    print("Concatenating input_ids...")
    concatenated_input_ids = tf.concat(input_ids, 0)
    del input_ids
    gc.collect()

    print("Concatenating attention_masks...")
    concatenated_attention_masks = tf.concat(attention_masks, 0)
    del attention_masks
    gc.collect()

    return concatenated_input_ids, concatenated_attention_masks


def main():
    train_x, train_y, val_x, val_y, _, _ = load_csv()

    train_samples = np.random.choice(len(train_x), size=100000, replace=False)
    train_x = train_x[train_samples]
    train_y = train_y[train_samples]

    val_samples = np.random.choice(len(train_x), size=10000, replace=False)
    val_x = val_x[val_samples]
    val_y = val_y[val_samples]

    input_ids_train, attention_masks_train = encode(train_x)
    input_ids_val, attention_masks_val = encode(val_x)

    labels_train = tf.convert_to_tensor(train_y)
    labels_val = tf.convert_to_tensor(val_y)

    model: tf.keras.Model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics="accuracy")

    model.summary()

    log_path = "logs.csv"

    import pandas as pd

    try:
        history_df = pd.read_csv(log_path)
        print("Model history loaded successfully!")
    except Exception as e:
        print("No model history found or error in loading. Building a new history.")
        print("Error:", e)
        # create empty df
        history_df = pd.DataFrame()

    model.fit(
        [input_ids_train, attention_masks_train],
        labels_train,
        validation_data=([input_ids_val, attention_masks_val], labels_val),
        callbacks=[
            CSVLogger(log_path, append=True),
        ],
        initial_epoch=history_df.shape[0],
        epochs=4,
        batch_size=6,
    )

    # reload history after training
    history_df = pd.read_csv(log_path)

    metrics = ["accuracy", "loss", "precision", "recall"]
    for metric in metrics:
        plt.plot(history_df[metric])
        plt.plot(history_df["val_" + metric])
        plt.title("Model " + metric)
        plt.ylabel(metric)
        plt.xlabel("Epoch")
        plt.legend(["Train", "Val"], loc="upper left")
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
