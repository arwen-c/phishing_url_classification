# ML

### Install RAPIDS

```shell
conda create --solver=libmamba -n rapids-23.12 -c rapidsai -c conda-forge -c nvidia cudf=23.12 cuml=23.12 python=3.10 cuda-version=12.0
conda activate rapids-23.12
conda install tldextract tqdm pymongo matplotlib
```

### Performance

Logistic Regression:
100%|██████████| 1000/1000 [02:42<00:00,  6.15trial/s, best loss: -0.8254219889640808]
Best params:  {'c': 59.643464530019735, 'class_weight': 1, 'fit_intercept': 0, 'l1_ratio': 0.20138101954231868, 'penalty': 1, 'tol': 9.246728713170994e-05}

Linear SVC:
100%|██████████| 1000/1000 [11:51<00:00,  1.41trial/s, best loss: -0.8253193497657776]
Best params:  {'c': 16.381478122844584, 'change_tol': 0.0006360348597608416, 'class_weight': 1, 'fit_intercept': 0, 'grad_tol': 0.00048694202862422303, 'loss': 1, 'penalized_intercept': 0, 'penalty': 0, 'tol': 11.881796519053097}

Naive Bayes:
100%|██████████| 1000/1000 [00:26<00:00, 37.10trial/s, best loss: -0.7135170698165894]
Best params:  {'alpha': 0.12695561035635183, 'fit_prior': 0}

Nearest Neigbors:
100%|██████████| 1000/1000 [4:08:21<0:00:00, 15.33s/trial, best loss: -0.9260850548744202]
Best params:  {'metric': 3, 'n_neighbors': 2}

Random Forest:
100%|██████████| 50/50 [02:55<00:00,  3.51s/trial, best loss: -0.7207251787185669]
Best params:  {'bootstrap': 1, 'max_depth': 0, 'max_samples': 0.24595877227188392, 'min_impurity_decrease': 0.09423017634336195, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_bins': 0, 'n_estimators': 1}
