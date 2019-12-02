# Titanic-Machine-Learning-from-Disaster
## Introduce:
The this project is using machine learning to create a model that predicts which passengers survived the Titanic shipwreck.
## Purpose
In this project, I will build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
## Architecture
### Titanic.csv: This is the data of Titanic event. 

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

### Titanic.py:
Module: sklearn, pandas and matplotlib.pyplot.<br>
Function: The code is used to creat a machine learning model to predict who will suvive. 

### analysisData.py
Module: seaborn, pandas and matplotlib.pyplot.<br>
Function: Analyzing the relationship between to different features and survived. Find out which feature has a big impact on survival.

### Pictures

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

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/tree1.png)

The graph of the tree before limit the max size

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/1_train.PNG)

The accuracy of prediction to training data is 96.94%(The tree before limit the max size)

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/1_test.PNG)

The accuracy of prediction to training data is 73.54%(The tree before limit the max size)

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/Figure_misclassificated.png)

The relationship between misclassificated and accuracy

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/best_tree.png)

The graph of the tree after limit the max size

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/besttreeTrain.PNG)

The accuracy of prediction to training data is 81.77%(The tree after limit the max size)

![](https://github.com/wzhh1995/Titanic-Machine-Learning-from-Disaster/blob/master/pictures/bestTreeTest.PNG)

The accuracy of prediction to testing data is 81.42%(The tree after limit the max size)

### Conclusion
First, using the overview of data can find that data of some features are not complete. Chosing age, sex, pclass, sibsp, parch, fare, cabin and embarked to analyze. Figuring out which feature influent the survival rates more. Then use these features (age, sex, fare, sibsp and parch) to create a machine-learning model, a decision tree. Prevent overfitting by limiting the maximum size(the best size is 3). Eventually reached an 81% accuracy rate to predict whether passengers survived.

## Todo
Find more methods such as limiting the maximum number of leaf nodes or using other machine learning models to increase the accuracy of predictions.

## Author
Zhihao Wang
