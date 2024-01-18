# ML

### Install RAPIDS

```shell
conda create --solver=libmamba -n rapids-23.12 -c rapidsai -c conda-forge -c nvidia cudf=23.12 cuml=23.12 python=3.10 cuda-version=12.0
conda activate rapids-23.12
conda install tldextract tqdm pymongo matplotlib
```

### Performance

Logistic Regression:
0.8423968553543091
Best params:  {'c': 70.24682387842596, 'class_weight': 1, 'fit_intercept': 0, 'l1_ratio': 0.8765034555778022, 'penalty': 0}

Linear SVC:
0.841742753982544
Best params:  {'c': 0.0010484883768061676, 'class_weight': 1, 'fit_intercept': 0, 'loss': 1, 'penalized_intercept': 0, 'penalty': 1}

Naive Bayes:
0.7152677774429321
Best params:  {'alpha': 1.1583084259806466}

Nearest Neigbors:
0.9478824734687805
Best params:  {'metric': 13, 'n_neighbors': 1}

Random Forest:
0.736885666847229
Best params:  {'bootstrap': 1, 'max_depth': 2, 'max_samples': 0.6407690956805958, 'min_samples_leaf': 1, 'min_samples_split': 0, 'n_bins': 0, 'n_estimators': 2}
