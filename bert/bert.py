import pandas as pd
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4028)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


def main():
    train_df = pd.read_csv("data/train_x.csv").sample(n=10000)
    val_df = pd.read_csv("data/val_x.csv").sample(n=1000)
    test_df = pd.read_csv("data/test_x.csv").sample(n=1000)

    X_train, y_train = list(train_df["train_x"]), list(train_df["train_y"])
    X_val, y_val = list(val_df["val_x"]), list(val_df["val_y"])
    X_test, y_test = list(test_df["test_x"]), list(test_df["test_y"])

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

    def encode(urls):
        input_ids = []
        attention_masks = []

        for url in urls:
            inputs = tokenizer.encode_plus(
                url,
                max_length=160,
                padding="max_length",
                return_tensors='tf',
            )
            input_ids.append(inputs['input_ids'])
            attention_masks.append(inputs['attention_mask'])

        return tf.concat(input_ids, 0), tf.concat(attention_masks, 0)

    input_ids_train, attention_masks_train = encode(X_train)
    input_ids_val, attention_masks_val = encode(X_val)
    input_ids_test, attention_masks_test = encode(X_test)

    labels_train = tf.convert_to_tensor(y_train)
    labels_val = tf.convert_to_tensor(y_val)
    labels_test = tf.convert_to_tensor(y_test)

    model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics="accuracy")

    batch_size = 6
    epochs = 3

    model.fit(
        [input_ids_train, attention_masks_train],
        labels_train,
        validation_data=([input_ids_val, attention_masks_val], labels_val),
        batch_size=batch_size,
        epochs=epochs
    )

    result = model.evaluate([input_ids_test, attention_masks_test], labels_test)
    print(f"Test Loss: {result[0]}")
    print(f"Test Accuracy: {result[1]}")


if __name__ == "__main__":
    main()
