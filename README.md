# Titanic-Machine-Learning-from-Disaster
## Introduce:
The this project is using machine learning to create a model that predicts which passengers survived the Titanic shipwreck.
## Purpose
In this project, I will build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
## Architecture
#### Titanic.csv: This is the data of Titanic event. 

pclass => Ticket class<br>
survived => Survival,0 = No, 1 = Yes<br>
name => Passenger's name<br>
sex => Genger<br>
age => Age in years<br>
sibsp => of siblings / spouses aboard the Titanic<br>
parch => of parents / children aboard the Titanic<br>
ticket => Ticket number<br>
fare => Passenger fare<br>
cabin => Cabin number<br>
embarked => Port of Embarkation; 	C = Cherbourg, Q = Queenstown, S = Southampton<br>
boat => Whether to board a lifeboat<br>

#### Titanic.py:
Module: sklearn, pandas and matplotlib.pyplot.<br>
Function: The code is used to creat a machine learning model to predict who will suvive. 

#### analysisData.py
Module: seaborn, pandas and matplotlib.pyplot.
Function: Analyzing the relationship between to different features and survived. Find out which feature has a big impact on survival.

#### Pictures

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/dataView.PNG)

Overview of the overall data


![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_age.png)

The relationship between age and survived


![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_gender.png)

The relationship between sex and survived

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_pclass.png)

The relationship between pclass and survived

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_familySize.png)

The relationship between sibsp, parch and survived

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_fare.png)

The relationship between fare and survived

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_cabin.png)

The relationship between cabin and survived

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_embarked.png)

The relationship between embarked and survived

## Create a decision tree to predict(The code of this process is "Titanic.py")

After the analysis of data, the next step is going to create a machine learning model. In this case, I will use the decision tree to predict the survived.

First, Import the data. And because of the analysis above, I will use the age, sex, pclass, fare, sibsp and parch, these six features, to create the model. 

```Python
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def decision():
    # Read the data of "Titanic.csv"
    data = pd.read_csv("Titanic.csv", encoding="ANSI")

    # Through analysis the data, using part of the data for decisionTree
    x = data[["pclass", "sex", "age", "fare", "parch", "sibsp"]]
    y = data["survived"]
```

Becasue the data of age and fare are not full, so we fill them with the mean number.

```Python
    # transfer the NaN in "age" and "fare" to the mean
    x["age"].fillna(x["age"].mean(), inplace=True)
    x["fare"].fillna(x["fare"].mean(), inplace=True)
```

Then split the data. The 70% data is used as training data and the 30% data is used as testing data.

```Python
    # Split the data, 70% become training data and 30% become testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
```

Feature engineering, transfer the data to one-hot code.

```Python
    dict = DictVectorizer()
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    x_test = dict.transform(x_test.to_dict(orient="records"))
```

Then, create the decision tree and generate the piction of the tree

```Python
    dec = DecisionTreeClassifier()
    dec.fit(x_train, y_train)
    export_graphviz(dec, out_file="tree1.dot", feature_names=dict.get_feature_names())
```

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/tree1.png)

show the accuracy of prediction to the training data and testing data

```Python
    # Evaluation the accuracy of results And print it out
    # show the accuracy of results of prediction the training data
    print("Training Data Score:", dec.score(x_train, y_train))
    report = classification_report(y_train, dec.predict(x_train))
    print(report)
    report = report.replace(" ", "")
    print("In-sample percent survivors correctly predicted:", report[54:58])
    print("In-sample percent fatalities correctly predicted:", report[37:41])
    print('\n\n')

    # show the accuracy of results of prediction the testing data
    print("Testing Data Score:", dec.score(x_test, y_test))
    report = classification_report(y_test, dec.predict(x_test))
    print(report)
    report = report.replace(" ", "")
    print("Out-of-sample percent survivors correctly predicted:", report[54:58])
    print("Out-of-sample percent fatalities correctly predicted:", report[37:41])
```

The accuracy of prediction to training data is 96.94%

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/1_train.PNG)

The accuracy of prediction to training data is 73.54%

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/1_test.PNG)

We can find out that the accuracy of prediction to training data is very high and to the testing data is low. This is because of the tree overfitting. I decide use limit the max size of the tree to fix this.

First, draw a graph to analysis the relationship between the tree size and the Misclassification.

```Python
    k_range = range(1, 50)
    cv_scores = []
    kmap = {}
    misclassification = []
    for n in k_range:
        dec = DecisionTreeClassifier(max_depth=n)
        scores = cross_val_score(dec, x_train, y_train, cv=10, scoring='accuracy')
        a = scores.mean()
        cv_scores.append(a)
        kmap[a] = n
    for i in cv_scores:
        misclassification.append((1 - i) * (y_train.size))
    plt.plot(k_range, misclassification)
    plt.xlabel('max_depth')
    plt.ylabel('Number of Misclassification')
    plt.show()
```


![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_misclassificated.png)

Then I can find the best size.

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/best%20size.PNG)

Use the best size to creat a new decision tree and generate the piction of the tree

```Python
    best_k = kmap[(max(cv_scores))]
    print("Best tree size is", best_k)
    best_dec = DecisionTreeClassifier(max_depth=best_k)
    best_dec.fit(x_train, y_train)
    export_graphviz(best_dec, out_file="best_tree.dot", feature_names=dict.get_feature_names())
```

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/best_tree.png)

At last show the accuracy of prediction to the training data and testing data.

```Python
    print('\n\noptimal tree:\n\n')
    print("Training Data Score:", best_dec.score(x_train, y_train))
    report = classification_report(y_train, best_dec.predict(x_train))
    print(report)
    report = report.replace(" ", "")
    print("In-sample percent survivors correctly predicted:", report[54:58])
    print("In-sample percent fatalities correctly predicted:", report[37:41])
    print('\n\n')
    print("Testing Data Score:", best_dec.score(x_test, y_test))
    report = classification_report(y_test, best_dec.predict(x_test))
    print(report)
    report = report.replace(" ", "")
    print("Out-of-sample percent survivors correctly predicted:", report[54:58])
    print("Out-of-sample percent fatalities correctly predicted:", report[37:41])
```

The accuracy of prediction to training data is 81.77%

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/besttreeTrain.PNG)

The accuracy of prediction to testing data is 81.42%

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/bestTreeTest.PNG)

At last, I got 81% accuracy of prediction better than the first tree.
