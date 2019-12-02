# Titanic-Machine-Learning-from-Disaster
## Introduce:
The this project is using machine learning to create a model that predicts which passengers survived the Titanic shipwreck.
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this project, I will build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
## Analyze The Data
The data of Titanic is in "Titanic.csv". The classifications in the data are "pclass", "survived", "name", "sex", "age", "sibsp", "parch"	"ticket", "fare", "cabin", "embarked", "boat", "body" and "home.dest".

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

Looking at the data set information, you can see that there are missing values for age, fare, cabin, embarked, boat, body and home.dest

```Python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
data = pd.read_csv("D:\PycharmProjects\Titanic\Titanic.csv",index_col=0)

# show the information of data
print(data.info())
```

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/dataView.PNG)

### Analyze the relationship between each feature and survived(This process is at "analysisData.py")

Use graphs to analyze the relationship between individual features and survival, so as to find out which parameters should be entered when building a machine learning model using a decision tree. According to the analysis above, I decided to analyze 7 features.

#### 1.AGE

Using data for observation and analysis, you can see that the last surviving age will be lower overall, but the characteristics are not obvious enough.

```Python
def age_survived():
    sns.boxplot(y=data["age"],x=data["survived"])
    plt.show()
```

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_age.png)


#### 2.Sex

Lady first. <br>
We can find that women account for a larger proportion of people rescued in the graph.

```Python
def gender_survived():
    Survived_m = data.survived[data.sex == 'male'].value_counts()
    Survived_f = data.survived[data.sex == 'female'].value_counts()
    df=pd.DataFrame({'male':Survived_m, 'female':Survived_f})
    df.plot(kind='bar', stacked=True)
    plt.xlabel("survived")
    plt.ylabel("number")
    plt.show()
```
![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_gender.png)

#### 3.Pclass

Chance to be rescued with cabin 1st is greater.

```Python
def pclass_survived():
    sns.barplot(x=data["pclass"],y=data["survived"])
    plt.show()
```
![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_pclass.png)

### 4.FamilySize

Is there a high or low chance of being rescued by a large number of families when disaster strikes?

Combine siblings and parents to get the total number of families.

It can be seen that the number of families is more likely to be rescued between 2-4 people, single people lack help, and more families need to take care of more families.

```Python
def familySize_survived():
    data["fsize"] = data["sibsp"] + data["parch"] + 1
    sns.barplot(x = data["fsize"],y = data["survived"])
    plt.show()
```

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_familySize.png)

#### 5.Fare

The chance of being rescued with high fares is relatively high, but it is not obvious.

```Python
def fare_survived():
    sns.boxplot(y=data["fare"],x=data["survived"])
    plt.show()
```

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_fare.png)

#### 6.Carbin

The cabin number represents the position on the ship. There may be a certain correlation between the probability of different cabin positions running to the rescue ship.

The missing cabin may be no cabin, and replaced with X.

With cabins overall there is a higher chance of being rescued than without cabins.

```Python
def cabin_survived():
    data["cabin"] = pd.DataFrame([i[0] if not pd.isnull(i) else "X" for i in data["cabin"]])
    data.cabin.unique()
    sns.barplot(x=data["cabin"], y=data["survived"])
    plt.show()
```

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_cabin.png)

#### 7.Embarked

It can be seen from the figure that the rescue chances between different boarding ports are not much different. After boarding, guests will basically go to their own cabins. It is speculated that different boarding ports will be divided into cabins or ticket types, but the Whether it was rescued has little effect.

```Python
def embarked_survived():
    sns.barplot(x=data["embarked"], y=data["survived"])
    plt.show()
```

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_embarked.png)

## Create a decision tree to predict

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
