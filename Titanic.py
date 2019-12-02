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

    # transfer the NaN in "age" and "fare" to the mean
    x["age"].fillna(x["age"].mean(), inplace=True)
    x["fare"].fillna(x["fare"].mean(), inplace=True)

    # Split the data, 70% become training data and 30% become testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # Feature engineering, one-hot encode
    dict = DictVectorizer()
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    x_test = dict.transform(x_test.to_dict(orient="records"))

    # create the decisionTree
    dec = DecisionTreeClassifier()
    dec.fit(x_train, y_train)
    export_graphviz(dec, out_file="tree1.dot", feature_names=dict.get_feature_names())

    # Evaluation the accuracy of results And print it out
    # show the accuracy of results of predicting the training data
    print("Training Data Score:", dec.score(x_train, y_train))
    report = classification_report(y_train, dec.predict(x_train))
    print(report)
    report = report.replace(" ", "")
    print("In-sample percent survivors correctly predicted:", report[54:58])
    print("In-sample percent fatalities correctly predicted:", report[37:41])
    print('\n\n')

    # show the accuracy of results of predicting the testing data
    print("Testing Data Score:", dec.score(x_test, y_test))
    report = classification_report(y_test, dec.predict(x_test))
    print(report)
    report = report.replace(" ", "")
    print("Out-of-sample percent survivors correctly predicted:", report[54:58])
    print("Out-of-sample percent fatalities correctly predicted:", report[37:41])

    # draw a graph to analysis the relationship between the tree size and the Misclassification
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

    # find the best tree size
    best_k = kmap[(max(cv_scores))]
    print("Best tree size is", best_k)

    best_dec = DecisionTreeClassifier(max_depth=best_k)
    best_dec.fit(x_train, y_train)
    export_graphviz(best_dec, out_file="best_tree.dot", feature_names=dict.get_feature_names())

    # Evaluation the accuracy of results And print it out
    # show the accuracy of results of predicting the training data
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

    return None


if __name__ == '__main__':
    decision()
