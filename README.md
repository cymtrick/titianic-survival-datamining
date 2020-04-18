```python
import numpy as np
import pandas as pandas
import re as re
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

train = pandas.read_csv('/Users/prashanthvarma/Downloads/train.csv', header = 0, dtype={'Age': np.float64})
test  = pandas.read_csv('/Users/prashanthvarma/Downloads/test.csv' , header = 0, dtype={'Age': np.float64})


print(train.info())   
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    None



```python
#  Fare and Survival mean cut to four categories
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
train['Fare'] = pandas.qcut(train['Fare'], 4)
print(train[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean())
```

                  Fare  Survived
    0   (-0.001, 7.91]  0.197309
    1   (7.91, 14.454]  0.303571
    2   (14.454, 31.0]  0.454955
    3  (31.0, 512.329]  0.581081



```python
# Embarked and Survival mean
train['Embarked'] = train['Embarked'].fillna('S')
print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
```


```python
# filling the empty ages and making mean with the survival rate
average_age = train['Age'].mean()
std_age = train['Age'].std()
null_count_age = train['Age'].isnull().sum()

null_random_list = np.random.randint(average_age - std_age, average_age + std_age , size = null_count_age)

train['Age'] = pandas.cut(train['Age'],5)

print (train[['Age', 'Survived']].groupby(['Age'], as_index=False).mean())

```

                    Age  Survived
    0    (0.34, 16.336]  0.550000
    1  (16.336, 32.252]  0.369942
    2  (32.252, 48.168]  0.404255
    3  (48.168, 64.084]  0.434783
    4    (64.084, 80.0]  0.090909



```python
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
```

       FamilySize  Survived
    0           1  0.303538
    1           2  0.552795
    2           3  0.578431
    3           4  0.724138
    4           5  0.200000
    5           6  0.136364
    6           7  0.333333
    7           8  0.000000
    8          11  0.000000



```python
train['alone'] = 0
train.loc[train['FamilySize'] == 1, 'alone'] = 1
print (train[['alone', 'Survived']].groupby(['alone'], as_index=False).mean())
```

       alone  Survived
    0      0  0.505650
    1      1  0.303538



```python
sns.heatmap(train.isnull())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a29aabc50>




![png](output_6_1.png)



```python
sns.countplot(x='Survived',hue='Sex',data=train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a29041390>




![png](output_7_1.png)



```python
sns.countplot(x='Survived',hue='Pclass',data=train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a28fdf810>




![png](output_8_1.png)



```python
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a29041190>




![png](output_9_1.png)



```python

```


```python

def clean_data(data):
 
    # Transforming the Age into {0,4}
    average_age = data['Age'].mean()
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    std_age = data['Age'].std()
    null_count_age = data['Age'].isnull().sum()
    null_random_list = np.random.randint(average_age - std_age, average_age + std_age , size = null_count_age)
    data['Age'][np.isnan(data['Age'])] = null_random_list
    data['Age'] = data['Age'].astype(int)
    data.loc[ data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age'] = 4
    
    
    # Adding the alone column as feature {0,1}
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['alone'] = 0
    data.loc[data['FamilySize'] == 1, 'alone'] = 1
    
    # Transforming the Embarked into {0,2}
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    data['Sex'] = data['Sex'].map({"female":0,"male":1})
    
    # Transforming the fare 10 {0,5}
#     (-0.001, 7.91]  0.197309
#     (7.91, 14.454]  0.303571
#     (14.454, 31.0]  0.454955
#     (31.0, 512.329]  0.581081
    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454),'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31.0),'Fare']   = 2
    data.loc[(data['Fare'] > 31.0) & (data['Fare'] <= 512.329),'Fare'] = 3
    data.loc[data['Fare'] >= 512.329, 'Fare'] = 5
    data['Fare'] = data['Fare'].astype(int)
    
    # Dropping the unnecessary columns
    drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch']
    data = data.drop(drop_columns, axis = 1)
    return data

train_data = pandas.read_csv('/Users/prashanthvarma/Downloads/train.csv', header = 0, dtype={'Age': np.float64})
test_data = pandas.read_csv('/Users/prashanthvarma/Downloads/test.csv', header = 0, dtype={'Age': np.float64})
train = clean_data(train_data)
test = clean_data(test_data)
print(train)
```

         Survived  Pclass  Sex  Age  Fare  Embarked  FamilySize  alone
    0           0       3    1    1     0         0           2      0
    1           1       1    0    2     3         1           2      0
    2           1       3    0    1     1         0           1      1
    3           1       1    0    2     3         0           2      0
    4           0       3    1    2     1         0           1      1
    ..        ...     ...  ...  ...   ...       ...         ...    ...
    886         0       2    1    1     1         0           1      1
    887         1       1    0    1     2         0           1      1
    888         0       3    0    2     2         0           4      0
    889         1       1    1    1     2         1           1      1
    890         0       3    1    1     0         2           1      1
    
    [891 rows x 8 columns]


    /Users/prashanthvarma/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      if __name__ == '__main__':



```python
#Classifier learning

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test
print(X_train.info())
print(X_test.info())

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 7 columns):
    Pclass        891 non-null int64
    Sex           891 non-null int64
    Age           891 non-null int64
    Fare          891 non-null int64
    Embarked      891 non-null int64
    FamilySize    891 non-null int64
    alone         891 non-null int64
    dtypes: int64(7)
    memory usage: 48.9 KB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 7 columns):
    Pclass        418 non-null int64
    Sex           418 non-null int64
    Age           418 non-null int64
    Fare          418 non-null int64
    Embarked      418 non-null int64
    FamilySize    418 non-null int64
    alone         418 non-null int64
    dtypes: int64(7)
    memory usage: 23.0 KB
    None



```python
classifiers = [LogisticRegression(),RandomForestClassifier(),Perceptron(),
               SGDClassifier(),DecisionTreeClassifier(),KNeighborsClassifier(),SVC(),LinearSVC(),GaussianNB(),GradientBoostingClassifier(),LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis()]
```


```python
class_strings = ['LogisticRegression','RandomForestClassifier','Perceptron',
               'SGDClassifier','DecisionTreeClassifier','NeighborsClassifier','SVC','LinearSVC',
                 'GaussianNB','GradientBoostingClassifier','LinearDiscriminantAnalysis','QuadraticDiscriminantAnalysis']

col_scores = ["Classifier", "Accuracy"]
scores_pd = pandas.DataFrame(columns=col_scores)
scores = []
for x in classifiers:
    name = x.__class__.__name__
    model = x
    model.fit(X_train,Y_train)
    predictions = model.predict(X_test)
    score = model.score(X_train, Y_train) * 100
    scores.append(score)


for z in range(12):
        data = pandas.DataFrame([[class_strings[z],int(scores[z])]], columns=col_scores)
        scores_pd = scores_pd.append(data)
print(scores_pd)
sns.barplot(x='Accuracy', y='Classifier', data=scores_pd, color='g')
```

    /Users/prashanthvarma/opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /Users/prashanthvarma/opt/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    /Users/prashanthvarma/opt/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/prashanthvarma/opt/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


                          Classifier Accuracy
    0             LogisticRegression       80
    0         RandomForestClassifier       87
    0                     Perceptron       71
    0                  SGDClassifier       79
    0         DecisionTreeClassifier       87
    0            NeighborsClassifier       83
    0                            SVC       83
    0                      LinearSVC       80
    0                     GaussianNB       77
    0     GradientBoostingClassifier       84
    0     LinearDiscriminantAnalysis       79
    0  QuadraticDiscriminantAnalysis       80





    <matplotlib.axes._subplots.AxesSubplot at 0x1a29e50a10>




![png](output_14_3.png)



```python
final_classifier =  DecisionTreeClassifier()
final_classifier.fit(X_train, Y_train)
result = final_classifier.predict(X_test)
print(result)
```

    [0 0 0 0 0 0 1 0 1 0 0 1 1 0 1 1 0 0 0 1 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0
     0 0 1 0 0 0 1 1 0 1 0 1 0 0 0 1 1 0 0 0 0 0 1 0 0 0 1 0 1 1 0 0 1 1 0 0 0
     1 0 0 1 0 1 1 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0
     1 1 1 1 0 0 1 0 1 1 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0
     0 0 1 0 0 0 0 0 1 1 1 0 1 1 1 0 0 0 0 0 1 0 0 0 1 0 0 1 1 0 1 1 0 0 1 0 1
     0 1 0 0 0 0 0 1 0 1 0 0 1 0 0 1 1 1 1 0 0 1 0 1 0 0 0 0 1 1 0 1 0 1 0 1 0
     1 0 0 1 0 1 0 0 0 1 0 0 1 0 0 0 1 1 0 1 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 1
     0 0 0 1 1 1 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 1 0 0 0 0
     1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 1 1 0 1 0 0 0 1 1 0
     1 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 1 1 0 0 0 1 0 1 0 0 0 0 1 1 0 1 0 0 0 1 1
     0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 1 0 1 0 0 1 0 1 0 0 1 0
     0 1 1 1 1 1 0 1 0 0 1]



```python

```


```python
remove_features = pandas.DataFrame({'feature':X_train.columns,'unneccessary_features':np.round(final_classifier.feature_importances_,3)})
remove_features = remove_features.sort_values('unneccessary_features',ascending=True).set_index('feature')
remove_features.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unneccessary_features</th>
    </tr>
    <tr>
      <th>feature</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>alone</td>
      <td>0.012</td>
    </tr>
    <tr>
      <td>Embarked</td>
      <td>0.043</td>
    </tr>
    <tr>
      <td>Fare</td>
      <td>0.069</td>
    </tr>
    <tr>
      <td>Age</td>
      <td>0.086</td>
    </tr>
    <tr>
      <td>FamilySize</td>
      <td>0.162</td>
    </tr>
    <tr>
      <td>Pclass</td>
      <td>0.168</td>
    </tr>
    <tr>
      <td>Sex</td>
      <td>0.461</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train = train.drop("Survived", axis=1)
X_train = X_train.drop("alone",axis=1)
X_train = X_train.drop("Embarked",axis=1)
Y_train = train["Survived"]
X_test  = test.drop("alone",axis=1)
X_test = X_test.drop("Embarked",axis=1)
print(X_train.info())
print(X_test.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 5 columns):
    Pclass        891 non-null int64
    Sex           891 non-null int64
    Age           891 non-null int64
    Fare          891 non-null int64
    FamilySize    891 non-null int64
    dtypes: int64(5)
    memory usage: 34.9 KB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 5 columns):
    Pclass        418 non-null int64
    Sex           418 non-null int64
    Age           418 non-null int64
    Fare          418 non-null int64
    FamilySize    418 non-null int64
    dtypes: int64(5)
    memory usage: 16.5 KB
    None



```python
final_classifier =  DecisionTreeClassifier()
final_classifier.fit(X_train, Y_train)
result = final_classifier.predict(X_test)
print(result)
```

    [0 0 0 0 0 0 1 0 1 1 0 1 1 0 1 1 0 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0 0 1 0 0 0
     0 0 1 0 0 0 1 1 0 1 0 1 0 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 0 1 1 0 0 0
     1 0 0 1 0 1 1 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0
     0 1 1 1 0 0 1 0 1 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0
     0 0 1 0 0 0 0 0 1 1 1 0 1 1 1 0 0 1 0 0 1 0 0 0 1 0 0 1 1 0 1 1 0 0 1 0 1
     0 1 0 0 0 0 0 1 0 1 0 1 1 0 0 0 1 0 1 0 0 0 0 1 0 0 0 0 1 1 0 1 0 1 0 1 0
     1 0 1 1 0 1 0 0 0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 0 1
     0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 1 0 0 0 0
     1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 0 1 0 0 0 1 1 0
     1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 1 0 0 0 1 0 1 0 0 1 0 1 1 0 1 0 0 0 1 0
     0 1 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 1 0 0 1 0 1 0 0 1 0 1 0 0 0 0
     0 1 1 0 1 1 0 1 0 0 1]



```python

```


```python

```


```python

```
