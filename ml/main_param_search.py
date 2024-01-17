import math

import cupy as cp
import numpy as np
from hyperopt import Trials, fmin, tpe
import matplotlib.pyplot as plt

from data.load.load import load_feature_vector
from models import *


def main():
    train_x, train_y, val_x, val_y, _, _ = load_feature_vector(k=26)

    train_x = train_x[:10000]
    train_y = train_y[:10000]

    # NOTE: convert to float when using kernel ridge
    train_x = scale_data(cp.asarray(train_x).astype(cp.float32))
    train_y = cp.asarray(train_y).astype(cp.float32)
    val_x = scale_data(cp.asarray(val_x).astype(cp.float32))
    val_y = cp.asarray(val_y).astype(cp.float32)

    # Define the trials
    trials = Trials()

    print("Starting Hyperparameter Tuning...")
    space = NEAREST_NEIGHBORS_CLASSIFICATION_PARAMS
    original_space = space.copy()

    def objective(new_params):
        _, params = trials.idxs_vals
        num_param_sets = len(next(iter(params.values())))
        params = [{key: params[key][i] for key in params} for i in range(num_param_sets)]

        for param in params:
            if new_params == param:
                return {'status': STATUS_FAIL}

        return train_nearest_neighbors_classification(
            train_x, train_y,
            val_x, val_y,
            new_params
        )

    # Example of tuning Logistic Regression Hyperparameters
    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )
    print("Best params: ", best_params)

    # lowercase the dictionary keys of space
    space = {k.lower(): v for k, v in original_space.items()}

    results = trials.results
    _, params = trials.idxs_vals
    num_param_sets = len(next(iter(params.values())))
    params = [{key: params[key][i] for key in params} for i in range(num_param_sets)]

    # plot the results using matplotlib
    losses = [-x['loss']  if 'loss' in x else None for x in results]
    maxx = max([x for x in losses if x is not None])

    # Number of subplots
    num_params = len(params[0])
    fig, axs = plt.subplots(num_params, 1, figsize=(10, 5 * num_params))

    for idx, (param_name, ax) in enumerate(zip(params[0], axs)):
        # Extract the values for the current parameter
        param_values = np.array([param[param_name] for param in params])

        if "log" in str(space[param_name]):
            param_values = [math.log(x) for x in param_values]

        # Normalize the parameter values to the range of [0, 1] for color mapping
        norm = plt.Normalize(vmin=min(param_values), vmax=max(param_values))

        # Create a scatter plot
        scatter = ax.scatter(range(len(losses)), losses, c=param_values, cmap='viridis', norm=norm)

        # Adding a colorbar
        fig.colorbar(scatter, ax=ax, orientation='vertical', label=f'Parameter {param_name}')

        # print maxx line
        ax.plot(range(len(losses)), [maxx] * len(losses), 'r--')
        # print text above max line with the max
        ax.text(0, maxx, f'Max: {maxx:.3f}', fontsize=12)

        ax.set_yscale('log')

        # Setting the title for each subplot
        ax.set_title(f'Loss evolution colored by {param_name}')
        ax.set_xlabel('Test number')
        ax.set_ylabel('Loss')
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
