# AI study note 0
This note is from SJSU CMPE257 lectured by professor Bernardo and self study.

The common steps to write machine learning project code:

## **1. before training**

1.1 load dataset. We need to load dataset into memory. We could use some built-in dataset in `sklearn` python library, or load data from some files using `pandas` python library. For example:

```python
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True) #sklearn

movies_df = pd.read_csv('movies.csv') #pandas
```

It should be noted that in the real world there is a step 0 which is collect data. Sometimes this is the hardest step.

1.2 data inspection. We could use some simple functions or method to do the inspection. For example:

```python
print(diabetes_X) #python built-in function
movies_df.head() #DataFrame class has a head() function
```

1.3 data preprocessing. the DataFrame class in `pandas` library has some built-in methods to do the preprocessing. For example:

```python
#Clean up NaN values from dataframes
title_mask = movies_df['title'].isna()
movies_df = movies_df.loc[title_mask == False]
```

In real world, after preprocessing, we need to do the inspection again to make sure that the data is prepared well.

## **2. during training**

2.1 dataset splitting (for supervised learning). We need to split the dataset into training part and testing part. Sometimes, we need to split it into three parts: training, validation and testing. We could do it manually, but we could use `train_test_split()` function in `sklearn.model_selection`. For example:

```python
diabetes_X_train, diabetes_X_test = train_test_split(df_X[['bmi']], test_size = 0.4, random_state=42)
diabetes_y_train, diabetes_y_test = train_test_split(df_y, test_size = 0.4,random_state=42)
```

This step should be skipped if we choose unsupervised learning model.

2.2 model selection. There are many built-in models in `sklearn`. For example:

```python
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=1)
```

In a typical scenario, we should choose carefully the hyperparameter of the model, it will magnificantly influence the result.

2.3 model training. In many cases, there are few lines to do the training, but the library will do a lot under the hood! For instance:

```python
forest_model.fit(train_X, train_y)
```

## **3. after training**

3.1 model testing (for supervised learning). We need to do the testing (or prediction) on the testing dataset.

```python
melb_preds = forest_model.predict(val_X)
```

3.2 model evaluation. We could use some indications (accuracy, score, or loss function, whatever, you name it) to judge the model. Underfitting or Overfitting would be normal issues. Or, we use some visualization method to "see" the result.

There are many indications, `mean_squared_error`, `r2_score`, `accuracy_score`, etc.

As for the visualization, `matplotlib.pyplot` library is our favorite.

Typically we need to tune our model, and do 2.2–3.2 again. Sometimes we even have to repeat the process from the beginning!