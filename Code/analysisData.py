import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
data = pd.read_csv("Titanic.csv",index_col=0)

# show the information of data
print(data.info())

# show the relationship between age and survived in graph
def age_survived():
    sns.boxplot(y=data["age"],x=data["survived"])
    plt.show()

# show the relationship between gender and survived in graph
def gender_survived():
    Survived_m = data.survived[data.sex == 'male'].value_counts()
    Survived_f = data.survived[data.sex == 'female'].value_counts()
    df=pd.DataFrame({'male':Survived_m, 'female':Survived_f})
    df.plot(kind='bar', stacked=True)
    plt.xlabel("survived")
    plt.ylabel("number")
    plt.show()

# show the relationship between pclass and survived in graph
def pclass_survived():
    sns.barplot(x=data["pclass"],y=data["survived"])
    plt.show()

# show the relationship between age and familySize in graph
def familySize_survived():
    data["fsize"] = data["sibsp"] + data["parch"] + 1
    sns.barplot(x = data["fsize"],y = data["survived"])
    plt.show()

# show the relationship between age and fare in graph
def fare_survived():
    sns.boxplot(y=data["fare"],x=data["survived"])
    plt.show()

# show the relationship between age and cabin in graph
def cabin_survived():
    data["cabin"] = pd.DataFrame([i[0] if not pd.isnull(i) else "X" for i in data["cabin"]])        # use "X" instead of the null in "cabin"
    data.cabin.unique()
    sns.barplot(x=data["cabin"], y=data["survived"])
    plt.show()

# show the relationship between age and embarked in graph
def embarked_survived():
    sns.barplot(x=data["embarked"], y=data["survived"])
    plt.show()

if __name__ == '__main__':
    age_survived()
    gender_survived()
    pclass_survived()
    familySize_survived()
    fare_survived()
    cabin_survived()
    embarked_survived()