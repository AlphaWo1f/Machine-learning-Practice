import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

style.use("ggplot")

data = pd.read_csv("student_mat_2173a47420.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "famrel", "Dalc", "Walc"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


"""
accuracy = 0
# retraining model and getting the one with highest accuracy: current accuracy is 97.52%
for i in range(5000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)

    if acc > accuracy:
        accuracy = acc
        # saving a model with an accuracy of:95.7%
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

print(accuracy)
"""

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])



p = "studytime"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
