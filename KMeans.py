
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
import util_mnist_reader
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.datasets import mnist
from coclust.evaluation.external import accuracy
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

#loadin the Fashion-MNIST dataset
X_train, y_train = util_mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = util_mnist_reader.load_mnist('data/fashion', kind='t10k')

#concatenating the dataset
x = np.concatenate((X_train,X_test))
y = np.concatenate((y_train,y_test))

#normalizing the data
x=x/255

#checking the shapes of the data
print(X_train.shape)
print(y_train.shape)
print(x.shape, y.shape)

#defining the number of clusters
n_clusters = len(np.unique(y))

kmeans = KMeans(n_clusters=10, init='random',
n_init=10, max_iter=200, 
tol=1e-04, random_state=0, algorithm = 'elkan')
#fitting the model
kmeans.fit(x)
print('SSE for clusters = ', '10' ,'is', kmeans.inertia_)


#test_loss, test_acc = kmeans.evaluate(test_images,  test_labels, verbose=2)
predictions = kmeans.predict(X_test)
np_predictions = np.argmax(predictions)


print(predictions)

print(y_test)

print(metrics.adjusted_rand_score(y_test,predictions))

print(nmi(y_test,predictions))

print(accuracy(y, kmeans.labels_))


# In[2]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, kmeans.labels_))

from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

indexes = linear_assignment(_make_cost_m(confusion_matrix(y, kmeans.labels_)))
js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
cm2 = confusion_matrix(y, kmeans.labels_)[:, js]
print(cm2)

