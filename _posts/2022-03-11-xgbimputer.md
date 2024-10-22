---
title: XGBImputer - Extreme Gradient Boosting Imputer
author: leonardo
date: 2022-03-11
categories: [Development]
tags: [Python, Machine Learning, XGBoost, Data Imputation, MissForest]
render_with_liquid: false
---
![Visits Badge](https://badges.pufler.dev/visits/leonardodepaula/2022-03-11-xgbimputer)

When studying Machine Learning I was interested about the problem of missing data. So I found the paper "MissForest—non-parametric missing value imputation for mixed-type data" written by Daniel J. Stekhoven and Peter Bühlmann:

>Abstract
>
>Motivation: Modern data acquisition based on high-throughput technology is often facing the problem of missing data. Algorithms commonly used in the analysis of such large-scale data >often depend on a complete set. Missing value imputation offers a solution to this problem. However, the majority of available imputation methods are restricted to one type of >variable only: continuous or categorical. For mixed-type data, the different types are usually handled separately. Therefore, these methods ignore possible relations between variable >types. We propose a non-parametric method which can cope with different types of variables simultaneously.
>
>Results: We compare several state of the art methods for the imputation of missing values. We propose and evaluate an iterative imputation method (missForest) based on a random >forest. By averaging over many unpruned classification or regression trees, random forest intrinsically constitutes a multiple imputation scheme. Using the built-in out-of-bag error >estimates of random forest, we are able to estimate the imputation error without the need of a test set. Evaluation is performed on multiple datasets coming from a diverse selection >of biological fields with artificially introduced missing values ranging from 10% to 30%. We show that missForest can successfully handle missing values, particularly in datasets >including different types of variables. In our comparative study, missForest outperforms other methods of imputation especially in data settings where complex interactions and >non-linear relations are suspected. The out-of-bag imputation error estimates of missForest prove to be adequate in all settings. Additionally, missForest exhibits attractive >computational efficiency and can cope with high-dimensional data.

Their take on how to impute missing data seemed to be very interesting, so I developed a Python Package that implemented it, but leveraging the robustness and predictive power of the XGBoost algorithm, released two years after the MissForest paper.

I hope it can help people studying Machine Learning. Pull requests are welcome.

<https://pypi.org/project/xgbimputer/>

# XGBImputer

XGBImputer is an effort to implement the concepts of the MissForest algorithm proposed by Daniel J. Stekhoven and Peter Bühlmann[1] in 2012, but leveraging the robustness and predictive power of the XGBoost[2] algorithm released in 2014.

The package also aims to simplify the process of imputing categorical values in a scikit-learn[3] compatible way.

## Installation

```bash
$ pip install xgbimputer
```

## Approach

Given a 2D array X with missing values, the imputer:

* 1 - counts the missing values in each column and arranges them in the ascending order;

* 2 - makes an initial guess for the missing values in X using the mean for numerical columns and the mode for the categorical columns;

* 3 - sorts the columns according to the amount of missing values, starting with the lowest amount;

* 4 - preprocesses all categorical columns with scikit-learn's OrdinalEncoder to get a purely numerical array;

* 5 - iterates over all columns with missing values in the order established on step 1;

  * 5.1 - selects the column in context on the iteration as the target;

  * 5.2 - one hot encodes all categorical columns other than the target;

  * 5.3 - fits the XGBoost algorithm (XGBClassifier for the categorical columns and XGBRegressor for the numeric columns) where the target column has no missing values;

  * 5.4 - predicts the missing values of the target column and replaces them on the X array;

  * 5.5 - calculates the stopping criterion (gamma) for the numerical and categorical columns identified as having missing data;

* 6 - repeats the process described in step 5 until the stopping criterion is met; and

* 7 - returns X with the imputed values.

## Example

```python
import pandas as pd
from xgbimputer import XGBImputer

df = pd.read_csv('titanic.csv')
df.head()
```

```
|    |   PassengerId |   Pclass | Name                                         | Sex    |   Age |   SibSp |   Parch |   Ticket |    Fare |   Cabin | Embarked   |
|---:|--------------:|---------:|:---------------------------------------------|:-------|------:|--------:|--------:|---------:|--------:|--------:|:-----------|
|  0 |           892 |        3 | Kelly, Mr. James                             | male   |  34.5 |       0 |       0 |   330911 |  7.8292 |     nan | Q          |
|  1 |           893 |        3 | Wilkes, Mrs. James (Ellen Needs)             | female |  47   |       1 |       0 |   363272 |  7      |     nan | S          |
|  2 |           894 |        2 | Myles, Mr. Thomas Francis                    | male   |  62   |       0 |       0 |   240276 |  9.6875 |     nan | Q          |
|  3 |           895 |        3 | Wirz, Mr. Albert                             | male   |  27   |       0 |       0 |   315154 |  8.6625 |     nan | S          |
|  4 |           896 |        3 | Hirvonen, Mrs. Alexander (Helga E Lindqvist) | female |  22   |       1 |       1 |  3101298 | 12.2875 |     nan | S          |
```

```python
df = df.drop(columns=['PassengerId', 'Name', 'Ticket'])
df.info()
```

```text
class 'pandas.core.frame.DataFrame'
RangeIndex: 418 entries, 0 to 417
Data columns (total 8 columns):
#   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   Pclass    418 non-null    int64  
 1   Sex       418 non-null    object 
 2   Age       332 non-null    float64
 3   SibSp     418 non-null    int64  
 4   Parch     418 non-null    int64  
 5   Fare      417 non-null    float64
 6   Cabin     91 non-null     object 
 7   Embarked  418 non-null    object 
dtypes: float64(2), int64(3), object(3)
memory usage: 26.2+ KB
```

```python
df_missing_data = pd.DataFrame(df.isna().sum().loc[df.isna().sum() > 0], columns=['missing_data_count'])
df_missing_data['missing_data_type'] = df.dtypes
df_missing_data['missing_data_percentage'] = df_missing_data['missing_data_count'] / len(df)
df_missing_data = df_missing_data.sort_values(by='missing_data_percentage', ascending=False)
df_missing_data
```

```text
|       |   missing_data_count | missing_data_type   |   missing_data_percentage |
|:------|---------------------:|:--------------------|--------------------------:|
| Cabin |                  327 | object              |                0.782297   |
| Age   |                   86 | float64             |                0.205742   |
| Fare  |                    1 | float64             |                0.00239234 |
```

```python
imputer = XGBImputer(categorical_features_index=[0,1,6,7], replace_categorical_values_back=True)
X = imputer.fit_transform(df)
```

```text
XGBImputer - Epoch: 1 | Categorical gamma: inf/274. | Numerical gamma: inf/0.0020067522
XGBImputer - Epoch: 2 | Categorical gamma: 274./0. | Numerical gamma: 0.0020067522/0.0000494584
XGBImputer - Epoch: 3 | Categorical gamma: 0./0. | Numerical gamma: 0.0000494584/0.
XGBImputer - Epoch: 4 | Categorical gamma: 0./0. | Numerical gamma: 0./0.
```

```python
type(X)
```

```
text
numpy.ndarray
```

```python
pd.DataFrame(X).head(15)
```

```text
|    |   0 | 1      |       2 |   3 |   4 |       5 | 6               | 7   |
|---:|----:|:-------|--------:|----:|----:|--------:|:----------------|:----|
|  0 |   3 | male   | 34.5    |   0 |   0 |  7.8292 | C78             | Q   |
|  1 |   3 | female | 47      |   1 |   0 |  7      | C23 C25 C27     | S   |
|  2 |   2 | male   | 62      |   0 |   0 |  9.6875 | C78             | Q   |
|  3 |   3 | male   | 27      |   0 |   0 |  8.6625 | C31             | S   |
|  4 |   3 | female | 22      |   1 |   1 | 12.2875 | C23 C25 C27     | S   |
|  5 |   3 | male   | 14      |   0 |   0 |  9.225  | C31             | S   |
|  6 |   3 | female | 30      |   0 |   0 |  7.6292 | C78             | Q   |
|  7 |   2 | male   | 26      |   1 |   1 | 29      | C31             | S   |
|  8 |   3 | female | 18      |   0 |   0 |  7.2292 | B57 B59 B63 B66 | C   |
|  9 |   3 | male   | 21      |   2 |   0 | 24.15   | C31             | S   |
| 10 |   3 | male   | 24.7614 |   0 |   0 |  7.8958 | C31             | S   |
| 11 |   1 | male   | 46      |   0 |   0 | 26      | C31             | S   |
| 12 |   1 | female | 23      |   1 |   0 | 82.2667 | B45             | S   |
| 13 |   2 | male   | 63      |   1 |   0 | 26      | C31             | S   |
| 14 |   1 | female | 47      |   1 |   0 | 61.175  | E31             | S   |
```

```python
imputer2 = XGBImputer(categorical_features_index=[0,1,6,7], replace_categorical_values_back=False)
X2 = imputer2.fit_transform(df)
```

```text
XGBImputer - Epoch: 1 | Categorical gamma: inf/274. | Numerical gamma: inf/0.0020067522
XGBImputer - Epoch: 2 | Categorical gamma: 274./0. | Numerical gamma: 0.0020067522/0.0000494584
XGBImputer - Epoch: 3 | Categorical gamma: 0./0. | Numerical gamma: 0.0000494584/0.
XGBImputer - Epoch: 4 | Categorical gamma: 0./0. | Numerical gamma: 0./0.
```

```python
pd.DataFrame(X2).head(15)
```

```text
|    |   0 |   1 |       2 |   3 |   4 |       5 |   6 |   7 |
|---:|----:|----:|--------:|----:|----:|--------:|----:|----:|
|  0 |   2 |   1 | 34.5    |   0 |   0 |  7.8292 |  41 |   1 |
|  1 |   2 |   0 | 47      |   1 |   0 |  7      |  28 |   2 |
|  2 |   1 |   1 | 62      |   0 |   0 |  9.6875 |  41 |   1 |
|  3 |   2 |   1 | 27      |   0 |   0 |  8.6625 |  30 |   2 |
|  4 |   2 |   0 | 22      |   1 |   1 | 12.2875 |  28 |   2 |
|  5 |   2 |   1 | 14      |   0 |   0 |  9.225  |  30 |   2 |
|  6 |   2 |   0 | 30      |   0 |   0 |  7.6292 |  41 |   1 |
|  7 |   1 |   1 | 26      |   1 |   1 | 29      |  30 |   2 |
|  8 |   2 |   0 | 18      |   0 |   0 |  7.2292 |  15 |   0 |
|  9 |   2 |   1 | 21      |   2 |   0 | 24.15   |  30 |   2 |
| 10 |   2 |   1 | 24.7614 |   0 |   0 |  7.8958 |  30 |   2 |
| 11 |   0 |   1 | 46      |   0 |   0 | 26      |  30 |   2 |
| 12 |   0 |   0 | 23      |   1 |   0 | 82.2667 |  12 |   2 |
| 13 |   1 |   1 | 63      |   1 |   0 | 26      |  30 |   2 |
| 14 |   0 |   0 | 47      |   1 |   0 | 61.175  |  60 |   2 |
```

## License

Licensed under an [Apache-2](https://github.com/leonardodepaula/xgbimputer/blob/master/LICENSE) license.

## References

* [1] [Daniel J. Stekhoven and Peter Bühlmann. "MissForest—non-parametric missing value imputation for mixed-type data."](https://academic.oup.com/bioinformatics/article/28/1/112/219101)

* [2] [XGBoost](https://xgboost.ai/)

* [3] [scikit-learn](https://scikit-learn.org/)