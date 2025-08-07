![](UTA-DataScience-Logo.png)

# Fetal Health Classification

This repository contains files that attempt to use tree-based models to determine the health status of a fetus based on the results of cardiotocograms from the 
["Fetal Health Classification"](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) Kaggle dataset.

## Overview
Maternal and child mortality are key indicators of societal development, with most of these deaths occurring in low-income environments. A cardiotocogram (CTG) is a test that records vital signs such as fetal heart rate and uterine contractions to assess the health of a fetus in utero. It is inexpensive and can be used to detect early signs of fetal distress, allowing for rapid intervention and the prevention of child and/or maternal death. The goal of this project is to train a machine learning model to determine whether a fetus is normal, suspect (suspected of being in distress), or pathological (in distress) based on measurements from CTGs. Several iterations of Random Forest and XGBoost models, combined with sampling techniques such as SMOTE (Synthetic Minority Oversampling Technique) and random oversampling, are evaluated for their multi-class classification performance. An XGBoost with optimized hyperparameters performed the best with a macro-averaged F1-score of 0.912.

## Summary of Work Done

### Data

* **Type:** Tabular
  * Input: A CSV file that contains CTG measurements such as fetal heart rate, uterine contractions, and fetal movement.
  * Output: A health status label as determined by an obstetrician indicating whether a fetus is healthy, suspect, or pathological.
* **Size:**
  * 2,126 rows and 22 features, file size: 228.72 kB
* **Instances:**
  * Training: 1,700 instances
  * Testing: 426 instances
  * Validation: Performed using cross-validation in Grid Search

#### Preprocessing / Clean Up

* **Feature Types:** The features histogram_tendency and fetal_health were converted from numerical to categorical features because they represent the skew of a CTG histogram and fetal health classifications, respectively.
* **Encoding:** The feature histogram_tendency was one-hot encoded into three separate features, each representing the skew of the histogram.
* **Outliers:** Features such as histogram_number_of_zeros and fetal_movement had a significant number of outliers, but all outliers were retained to preserve any potential patterns that may correlate with the target class. 

#### Data Visualization

As noted in the notebooks, there is a significant class imbalance in the dataset:
* 77.8% Normal
* 13.9% Suspect
* 8.3% Pathological

The main focus of the data visualizations is to identify features that effectively separate the minority classes from the majority class.

![image](https://github.com/user-attachments/assets/2e9cf5a2-52bd-410c-8e9a-2d50a3b0c0e3)
![image](https://github.com/user-attachments/assets/36408af2-8bf7-4996-b237-ee18eb67721b)

Histogram_tendency represents the skewness of the fetal heart rate distribution during the CTG, while histogram_number_of_zeros shows the number of zero count bins in the histogram. The table shows that a left skew (-1.0) is more common in pathological cases: 27% of pathological cases have a left skew compared to less than 6% of normal or suspect cases. The graph shows that the majority of pathological class has no empty bins in their CTG histograms. 

![image](https://github.com/user-attachments/assets/a7d1a6c3-dbcd-4551-a3f1-9bce11d5a401)
![image](https://github.com/user-attachments/assets/77e22437-3dd9-44a8-9156-3045c615f54f)

Histogram_variance captures the variance of the fetal heart rate distribution during the CTG, while histogram_number_of_peaks indicates the number of distinct peaks. The data shows that almost all instances in the suspect class cluster near zero in histogram_variance. A large portion of the suspect class falls within the 0–5 range in histogram_number_of_peaks.

#### Problem Formulation
  * **Input:** `fetal_health.csv`, 22 features
  * **Output:** 1 = Normal, 2 = Suspect, 3 = Pathological
  * **Models:**
    * Random Forest: Serves as a baseline and is easily interpretable.  
    * XGBoost: Performs better than Random Forest, handles class imbalances and outliers well.
  * **Hyperparameters:**
     * Most of the models used default hyperparameters when being trained, except for class_weight='balanced' on Random Forest and sample_weight=sample_weights on XGBoost when SMOTE or Random Oversampling weren't being used. In addition, all XGBoost models had the hyperparameter objective='multi:softmax' set for multi-class classification. The optimized hyperparameters for the best performing XGBoost model were:
       * objective='multi:softmax', sample_weight=sample_weights, random_state=42, gamma=0.1, learning_rate=0.2, max_delta_step=1, max_depth=7, n_estimators=100

### Training

After the initial baseline model, the target column was label encoded for the XGBoost models. Multiple XGBoost models were trained using default hyperparameters, SMOTE, or Random Oversampling and then compared using macro-averaged F1-score and confusion matrices. Grid Search was then used on the best performing model. The main issue during training was that the suspect class had noticeably lower F1-scores and the confusion matrices showed that it was mislabeling instances more often than the other two classes. This is likely due to the suspect class often overlapping with the other classes in the data, resulting in poor separation between the classes.

### Performance Comparison

![image](https://github.com/user-attachments/assets/83d1ff5a-2b23-4a6a-89e4-66347a0698b5)

One of the main metrics I used to evaluate the models was macro-averaged F1-score. It gives equal weight to each class, preventing the normal class from skewing the score and giving the false impression of better model performance. As seen in the table, the sampling techniques had little effect on model performance. The best performing model didn't use any sampling techniques and only had hyperparameter tuning applied.

![image](https://github.com/user-attachments/assets/5c4148fa-0d05-4245-be8c-e79ab1623428)

The other main metric I used was a confusion matrix to see a visual representation of how each model classified each case. The confusion matrix above is the one for the optimized XGBoost model and shows that the model is correctly classifying each instance in the test set most of the time, except for a slightly worse performance in the suspect class. 

### Conclusions

An XGBoost model with optimized hyperparameters was the most effective at determining fetal health status. 

### Future Work

* **Different Sampling Techniques:** Explore other sampling methods when trying to address the class imbalance. Try to undersample the majority class as opposed to oversampling the minority ones or combine several of these methods together. 
* **Feature Engineering:** The models still struggled at separating the suspect class from the others due to the data for it often overlapping with them. Creating new features that highly correlate with the suspect class or removing existing features based on multicollinearity could help boost the metrics of the class. 
* **Outliers:** Although I chose to leave outliers alone, imputing outliers with other values such as the mean, median, mode of a column could give a better performance. However, I'd be cautious doing this since the data comes from a real life study and it can introduce unnecessary noise to the dataset.
* **Normalization:** Even though tree-based models were used, normalizing the data could still improve performance by highlighting existing outliers in the dataset.

## How to Reproduce Results

After downloading the CSV file, run the EDA and Preprocessing notebook to produce a preprocessed CSV file to use for training. Then, running the Training and Evaluations notebook will train all of the models and display the evaluations for each. 

### Overview of Files in Repository

* EDA and Preprocessing.ipynb: Contains data visualizations and summary info about the dataset. Produces a cleaned CSV file for training when run.
* Training and Evaluations.ipynb: Trains and evaluates several models using the cleaned dataset from the first notebook. 
* fetal_health.csv: The original dataset as provided by Kaggle.
  
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

Evaluation functions generate classification reports and confusion matrices for each model in the Training and Evaluations notebook. The table comparing the F1-scores of each model is also at the end of this notebook.

#### Citations
SisPorto 2.0: a program for automated analysis of cardiotocograms [(DOI: 10.48550/arXiv.2207.08815)](https://arxiv.org/abs/2207.08815)
