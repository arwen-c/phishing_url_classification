import sys
from models import feature_vector_models, string_embedding_models
from keras import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, CSVLogger
from matplotlib import pyplot as plt


def main():
    if len(sys.argv) < 3:
        print(
            "Please provide an input type, model name, and optionally a top k for the feature vector model."
        )
        print(
            f"python main.py ('string_embedding'/'feature_vector') ({list(string_embedding_models.keys())}/{list(feature_vector_models.keys())}) [k]"
        )
        exit(1)

    model_type = sys.argv[1].lower()
    model_name = sys.argv[2].lower()
    model_builder, load, transform_x, transform_y = None, None, None, None
    if model_type == "feature_vector":
        model_builder, load, transform_x, transform_y = feature_vector_models[
            model_name
        ]
    elif model_type == "string_embedding":
        model_builder, load, transform_x, transform_y = string_embedding_models[
            model_name
        ]
    else:
        print("Please provide a valid model type.")
        exit(1)

    top_k = 23
    if len(sys.argv) == 4:
        top_k = int(sys.argv[3])

    # Step 1: Load data
    train_x, train_y, val_x, val_y, test_x, test_y = load(k=top_k)

    train_x = transform_x(train_x)
    train_y = transform_y(train_y)
    val_x = transform_x(val_x)
    val_y = transform_y(val_y)
    test_x = transform_x(test_x)
    test_y = transform_y(test_y)

    prefix = f"{model_type}_{model_name}_{top_k}/"

    checkpoint_path = prefix + "cp.model.keras"
    checkpoint_path_loss = prefix + "cp.loss.model.keras"
    checkpoint_path_accuracy = prefix + "cp.accuracy.model.keras"

    try:
        model = load_model(checkpoint_path)
        print("Model loaded successfully!")
    except Exception as e:
        print("No model found or error in loading. Building a new model.")
        print("Error:", e)
        model: Model = model_builder(number_of_features=top_k)

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
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        callbacks=[
            ModelCheckpoint(checkpoint_path),
            ModelCheckpoint(
                checkpoint_path_loss,
                save_best_only=True,
                monitor="val_loss",
                mode="min",
            ),
            ModelCheckpoint(
                checkpoint_path_accuracy,
                save_best_only=True,
                monitor="val_accuracy",
                mode="max",
            ),
            CSVLogger(log_path, append=True),
        ],
        initial_epoch=history_df.shape[0],
        epochs=30,
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
