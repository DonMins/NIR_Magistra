# import numpy as np
# from sklearn.svm import SVC
# import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions
# import sklearn.model_selection
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestClassifier
#
# from sklearn.neighbors import KNeighborsClassifier
#
#
# path1 = "Признаки\\3\\" + "allVall.txt"
# x = np.array(np.loadtxt(path1, delimiter=" "))
# print(x.shape)
#
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
#     x[:,0:30], x[:,30], test_size = 0.1, random_state = 0
# )
#
# pca = PCA(n_components = 2)
# X_train2 = pca.fit_transform(X_train)
# # X_test = pca.fit_transform(X_test)
#
#
# y_train = np.array(list(np.uint8(y_train)))
# y_test = np.array(list(np.uint8(y_test)))
#
# clf = SVC( gamma='auto')
# clf2 = SVC( gamma='auto')
# # clf = RandomForestClassifier(random_state=1, n_estimators=100)
# # clf = RandomForestClassifier(random_state=1, n_estimators=100)
# # clf2 = RandomForestClassifier(random_state=1, n_estimators=100)
#
#
# clf.fit(X_train2, y_train)
# clf2.fit(X_train, y_train)
#
# test_predictions = clf2.predict(X_test)
# test_predictions2 = clf2.predict(X_train)
#
# print("accuracy = ", sklearn.metrics.accuracy_score( test_predictions, y_test ))
# print("accuracyTrain = ", sklearn.metrics.accuracy_score( test_predictions2, y_train ))
#
# # a1 = plt.scatter(x[0:59,0],x[0:59,1],marker='s')
# # a2 = plt.scatter(x[60:121,0],x[60:121,1],marker='^')
#
#
# ax=plot_decision_regions(X_train2, y_train, clf=clf, legend=1)
# # Adding axes annotations
#
# plt.title('SVC')
#
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles,
#           ['Здоровые', 'Депрессия','Агрессия', 'Алкоголики'],
#            framealpha=0.3, scatterpoints=1)
#
# plt.show()



import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import sklearn.model_selection
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical


class Onehot2Int(object):

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        y_pred = self.model.predict(X)
        return np.argmax(y_pred, axis=1)

import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.data import iris_data
from mlxtend.preprocessing import standardize
from mlxtend.plotting import plot_decision_regions
from keras.utils import to_categorical

path1 = "Признаки\\3\\" + "allVall.txt"
x = np.array(np.loadtxt(path1, delimiter=" "))

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x[:,0:30], x[:,30], test_size = 0.3, random_state = 0
)

pca = PCA(n_components = 2)
X = pca.fit_transform(X_train)


y = np.array(list(np.uint8(y_train)))
y_test2 = np.array(list(np.uint8(y_test)))

# OneHot encoding
y_onehot = to_categorical(y)
y_test = to_categorical(y_test2)


np.random.seed(123)
model = Sequential()
model.add(Dense(2000, input_shape=(2,), activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1000, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(4, activation='softmax'))

# Configure the model and start training
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
history2 = model.fit(X, y_onehot, epochs=15, batch_size=5, verbose=1)


# Wrap keras model
model_no_ohe = Onehot2Int(model)

# Plot decision boundary
ax = plot_decision_regions(X, y, clf=model_no_ohe)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,
          ['Здоровые', 'Депрессия','Агрессия', 'Алкоголики'],
           framealpha=0.3, scatterpoints=1)
plt.show()






#
#
#
# # import numpy as np
# # from sklearn.svm import SVC
# # import matplotlib.pyplot as plt
# # from mlxtend.plotting import plot_decision_regions
# # import sklearn.model_selection
# # from sklearn.decomposition import PCA
# # from sklearn.ensemble import RandomForestClassifier
# #
# # from sklearn.neighbors import KNeighborsClassifier
# #
# #
# # path1 = "Признаки\\3\\" + "allVall.txt"
# # x = np.array(np.loadtxt(path1, delimiter=" "))
# # print(x.shape)
# #
# # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
# #     x[:,0:30], x[:,30], test_size = 0.1, random_state = 0
# # )
# #
# # pca = PCA(n_components = 2)
# # X_train2 = pca.fit_transform(X_train)
# # # X_test = pca.fit_transform(X_test)
# #
# #
# # y_train = np.array(list(np.uint8(y_train)))
# # y_test = np.array(list(np.uint8(y_test)))
# #
# # clf = SVC( gamma='auto')
# # clf2 = SVC( gamma='auto')
# # # clf = RandomForestClassifier(random_state=1, n_estimators=100)
# # # clf = RandomForestClassifier(random_state=1, n_estimators=100)
# # # clf2 = RandomForestClassifier(random_state=1, n_estimators=100)
# #
# #
# # clf.fit(X_train2, y_train)
# # clf2.fit(X_train, y_train)
# #
# # test_predictions = clf2.predict(X_test)
# # test_predictions2 = clf2.predict(X_train)
# #
# # print("accuracy = ", sklearn.metrics.accuracy_score( test_predictions, y_test ))
# # print("accuracyTrain = ", sklearn.metrics.accuracy_score( test_predictions2, y_train ))
# #
# # # a1 = plt.scatter(x[0:59,0],x[0:59,1],marker='s')
# # # a2 = plt.scatter(x[60:121,0],x[60:121,1],marker='^')
# #
# #
# # ax=plot_decision_regions(X_train2, y_train, clf=clf, legend=1)
# # # Adding axes annotations
# #
# # plt.title('SVC')
# #
# # handles, labels = ax.get_legend_handles_labels()
# # ax.legend(handles,
# #           ['Здоровые', 'Депрессия','Агрессия', 'Алкоголики'],
# #            framealpha=0.3, scatterpoints=1)
# #
# # plt.show()
#
#
#
# import numpy as np
# from sklearn.svm import SVC
# import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions
# import sklearn.model_selection
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestClassifier
#
# from sklearn.neighbors import KNeighborsClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.utils import to_categorical
#
#
# class Onehot2Int(object):
#
#     def __init__(self, model):
#         self.model = model
#
#     def predict(self, X):
#         y_pred = self.model.predict(X)
#         return np.argmax(y_pred, axis=1)
#
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# import matplotlib.pyplot as plt
# import numpy as np
# from mlxtend.data import iris_data
# from mlxtend.preprocessing import standardize
# from mlxtend.plotting import plot_decision_regions
# from keras.utils import to_categorical
#
# path1 = "Признаки\\3\\" + "allVall.txt"
# x = np.array(np.loadtxt(path1, delimiter=" "))
#
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
#     x[:,0:30], x[:,30], test_size = 0.3, random_state = 0
# )
#
# pca = PCA(n_components = 2)
# X = pca.fit_transform(X_train)
#
#
# y = np.array(list(np.uint8(y_train)))
# y_test2 = np.array(list(np.uint8(y_test)))
#
# # OneHot encoding
# y_onehot = to_categorical(y)
# y_test = to_categorical(y_test2)
#
#
# # Create the model
# # np.random.seed(123)
# # model = Sequential()
# # model.add(Dense(2000, input_shape=(2,), activation='relu', kernel_initializer='he_uniform'))
# # model.add(Dense(1000, activation='relu', kernel_initializer='he_uniform'))
# # model.add(Dense(4, activation='softmax'))
# #
# # # Configure the model and start training
# # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
# # history = model.fit(X, y_onehot, epochs=10, batch_size=5, verbose=1)
#
# np.random.seed(123)
# model2 = Sequential()
# model2.add(Dense(2000, input_shape=(30,), activation='relu', kernel_initializer='he_uniform'))
# model2.add(Dense(1000, activation='relu', kernel_initializer='he_uniform'))
# model2.add(Dense(4, activation='softmax'))
#
# # Configure the model and start training
# model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
# history2 = model2.fit(X_train, y_onehot, epochs=15, batch_size=5, verbose=1)
#
# test_predictions = model2.predict(X_test)
#
# print("accuracy = ", sklearn.metrics.accuracy_score( np.argmax(test_predictions, axis=1),y_test2))
#
# # # Wrap keras model
# # model_no_ohe = Onehot2Int(model)
# #
# # # Plot decision boundary
# # ax = plot_decision_regions(X, y, clf=model_no_ohe)
# # handles, labels = ax.get_legend_handles_labels()
# # ax.legend(handles,
# #           ['Здоровые', 'Депрессия','Агрессия', 'Алкоголики'],
# #            framealpha=0.3, scatterpoints=1)
# # plt.show()
