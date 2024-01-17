import tensorflow as tf
from tqdm import tqdm
from transformers import TFBertForSequenceClassification, BertTokenizer

from data.load.load import load_csv

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')


def encode(urls):
    input_ids = []
    attention_masks = []

    for url in tqdm(urls):
        inputs = tokenizer.encode_plus(
            url,
            max_length=160,
            padding="max_length",
            return_tensors='tf',
        )
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])

    return tf.concat(input_ids, 0), tf.concat(attention_masks, 0)


def main():
    train_x, train_y, val_x, val_y, test_x, test_y = load_csv()

    input_ids_train, attention_masks_train = encode(train_x)
    input_ids_val, attention_masks_val = encode(val_x)
    input_ids_test, attention_masks_test = encode(test_x)

    labels_train = tf.convert_to_tensor(train_y)
    labels_val = tf.convert_to_tensor(val_y)
    labels_test = tf.convert_to_tensor(test_y)

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
