# ICR---Identifying-Age-Related-Conditions
The code for my first Kaggle competition:

https://www.kaggle.com/competitions/icr-identify-age-related-conditions

in which I got 145th place and a silver medal.


The Optuna_Main.py is for running the hyperparameter tuning of the ensemble using Optuna and finding the best hyperparameters.
Then the Send_Main.py is for training the model with the chosen hyperparameters and producing the submission.csv file.
The icr_lib.py contains the code being used both by Optuna_Main.py and Send_Main.py.

My model is an ensemble of 3 enhanced-tree models: catboost, xgboost, lgbm and 2 tabPFNs (Deep neural network). The models are being tunned with consideration to minimazing the balanced log loss. additional sample_weignting and oversampling mechanisms were provided to fight the class imbalance in the training set.
