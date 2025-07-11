![](UTA-DataScience-Logo.png)

# Fetal Health Classification

This repository holds files to build an XGBoost model to determine the health status of a fetus based on cardiotocogram (CTG) results from the 
["Fetal Health Classification"](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) Kaggle dataset.

## Overview

* **Challenge:** Use features extracted from CTGs, such as uterine contractions and fetal heart rate, to determine whether a fetus is healthy, suspect, or pathological. 
* **Approach:** An XGBoost model using sample weights and optimized hyperparameters.
* **Summary:** The final XGBoost model achieved a macro-averaged F1-score of 0.912.

## Summary of Work Done

### Data

* **Type:** Tabular
* **Input:**
  * `fetal_health.csv`: CTG features with the health status labels determined by an obstetrician.
* **Size:**
  * 2,126 rows, 22 features, size: 228.72 kB
* **Instances (Train, Validation, Test Split):** 1,700 training, 426 test

#### Preprocessing / Clean Up

* Converted `histogram_tendency` and `fetal_health` to categorical features to maintain consistency with documentation.
* One-hot encoded `histogram_tendency`.

#### Data Visualization

There is a significant class imbalance in the dataset: 77.8% Normal, 13.9% Suspect, and 8.3% Pathological.

![image](https://github.com/user-attachments/assets/2e9cf5a2-52bd-410c-8e9a-2d50a3b0c0e3)
![image](https://github.com/user-attachments/assets/a7d1a6c3-dbcd-4551-a3f1-9bce11d5a401)

Features such as the ones above peformed well at separating the minority classes. Almost all features contributed to some extent in class separation.

### Problem Formulation

* **Define:**
  * **Input:** `fetal_health.csv`
  * **Output:** `df_encoded.csv` and classification model
  * **Model:**
    * XGBoost: robust to outliers, no scaling needed, handles class imbalances well, and strong multi-class classification performance.
    * Tuned Hyperparameters: `objective='multi:softmax'`, `random_state=42`, `gamma=0.1`, `learning_rate=0.2`, `max_delta_step=1`, `max_depth=7`, `n_estimators=100`

### Training
Although the original baseline model was a random forest, I ultimately chose an XGBoost model as the final model due to its better performance.  The training steps below describe the final XGBoost model workflow:

* Loaded the processed CSV file created by **EDA and Preprocessing**.
* Label-encoded the `fetal_health` column to be 0, 1, 2.
* Split the data into training and test sets.
* Calculated the sample weights for each class and incorporated them into the baseline model.
* Performed hyperparameter tuning using GridSearch.
* Create a new XGBoost model with optimized hyperparameters and evaluated it on the test set.

### Performance Comparison

![image](https://github.com/user-attachments/assets/83d1ff5a-2b23-4a6a-89e4-66347a0698b5)
![image](https://github.com/user-attachments/assets/5c4148fa-0d05-4245-be8c-e79ab1623428)

The macro-averaged F1-score is calculated by averaging the F1-score of each class. This metric was chosen because it gives equal weight to all classes, preventing the majority class from having too much influence on the final evaluation score.

### Conclusions

* XGBoost performs well on tabular data.

### Future Work

* Try undersampling the majority class instead of oversampling the minority ones.
* Perform PCA.
* Try other models like neural networks.

## How to Reproduce Results

Running all cells in **EDA and Preprocessing** will produce a processed CSV file. Running all cells in **Training and Evaluations** will generate the final model.

### Overview of Files in Repository

* `EDA and Preprocessing.ipynb`: Performs feature visualizations and outputs a clean CSV file.
* `Training and Evaluations.ipynb`: Trains and evaluates the models. Contains the evaluations for each. 
* `fetal_health.csv`: The original dataset as provided by kaggle.

### Required Libraries

* pandas
* numpy
* matplotlib
* ipython
* tabulate
* scikit-learn
* xgboost
* imbalanced-learn

#### Performance Evaluation

Evaluation functions generate classification reports and confusion matrices for each model in the **Training and Evaluations** notebook. The performance comparison table above is at the end of this notebook.

#### Citations
SisPorto 2.0: a program for automated analysis of cardiotocograms [(DOI: 10.48550/arXiv.2207.08815)](https://arxiv.org/abs/2207.08815)
