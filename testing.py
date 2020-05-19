from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

data_X = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'pental_length', 'pental_width'])
data_y = pd.DataFrame(y, columns=['training_class'])

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')