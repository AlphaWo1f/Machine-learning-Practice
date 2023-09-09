import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Get data from the text file and keep the attributes of interest
data = pd.read_csv("winequality_white.csv", sep=";")
data = data[["volatile acidity", "residual sugar", "chlorides", "free sulfur dioxide", "sulphates", "alcohol", "pH", "quality"]]

predict = "quality"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
score = linear.score(x_test, y_test)


predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print("prediction: ", predictions[x], "final: ", y_test[x])
print(score)



p = "sulphates"
style.use("ggplot")
pyplot.scatter(data[p], data["quality"])
pyplot.xlabel(p)
pyplot.ylabel("quality")
pyplot.show()