import time

import cupy as cp
import matplotlib.pyplot as plt

from data.load.load import load_feature_vector
from models import *

unscaled = ["train_naive_bayes", "train_stochastic_gradient_descent"]


def main():
    f = train_naive_bayes

    # warm the GPU:
    train_x, train_y, val_x, val_y, _, _ = load_feature_vector(k=26)

    train_x = cp.asarray(train_x).astype(cp.float32)
    train_y = cp.asarray(train_y).astype(cp.float32)
    val_x = cp.asarray(val_x).astype(cp.float32)
    val_y = cp.asarray(val_y).astype(cp.float32)

    if f.__name__ not in unscaled:
        train_x = scale_data(train_x)
        val_x = scale_data(val_x)

    # ignore the results
    print("warm 1")
    print(f(train_x, train_y, val_x, val_y, {}))
    print("warm 2")
    print(f(train_x, train_y, val_x, val_y, {}))
    print("warm 3")
    print(f(train_x, train_y, val_x, val_y, {}))

    ks = [1, 2, 3, 4, 5, 10, 15, 20, 26]
    scores = {"chi2": [], "mutual_info_classif": []}
    times = {"chi2": [], "mutual_info_classif": []}
    for k in ks:
        for score_func in scores.keys():
            train_x, train_y, val_x, val_y, _, _ = load_feature_vector(k=k, score_func=score_func)

            train_x = cp.asarray(train_x).astype(cp.float32)
            train_y = cp.asarray(train_y).astype(cp.float32)
            val_x = cp.asarray(val_x).astype(cp.float32)
            val_y = cp.asarray(val_y).astype(cp.float32)

            if f.__name__ not in unscaled and k > 1:
                train_x = scale_data(train_x)
                val_x = scale_data(val_x)

            # capture start time
            start = time.time()
            result = f(train_x, train_y, val_x, val_y, {})
            time_took = time.time() - start
            if result['status'] == STATUS_FAIL:
                print("Failed to train")
                return
            print("Score: ", -result['loss'])
            print("Time took: ", time_took)
            scores[score_func].append(-result['loss'])
            times[score_func].append(time_took)

    # Plot the accuracy results
    line1, = plt.plot(ks, scores["chi2"], '*-', label="Accuracy chi2")
    line2, = plt.plot(ks, scores["mutual_info_classif"], '*-', label="Accuracy information gain")
    # plot the maximum accuracy
    maxx = max(max(scores["chi2"]), max(scores["mutual_info_classif"]))
    plt.plot([0, 26], [maxx, maxx], 'k--')

    plt.text(0, maxx, f'Max: {maxx:.3f}', fontsize=12)

    plt.xlabel("k")
    plt.ylabel("Accuracy")

    # Use right axis for time
    ax2 = plt.twinx()
    line3, = ax2.plot(ks, times["chi2"], 'r*-', label="Time chi2")
    line4, = ax2.plot(ks, times["mutual_info_classif"], 'r*-', label="Time information gain")
    ax2.set_ylabel("Time (s)")

    # Create combined legend
    lines = [line1, line2, line3, line4]
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc='lower right')

    plt.title("Accuracy vs. k")
    plt.show()


if __name__ == "__main__":
    main()
