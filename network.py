import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.utils import shuffle
from kerastuner import RandomSearch, Hyperband, BayesianOptimization

def evaluate(P, T):
    # P - predictions
    # T - targets
    # accuracy = correct predictions / all predictions
    accuracy = np.mean(P == T)
    print("Точность = ", accuracy*100, '%')

path1 = "EEG_Features\\" + str(1) + ".txt"

X_train1 = np.loadtxt(path1, delimiter = " ")
X_trainAll = shuffle(X_train1)

X_train1 = X_trainAll[0:7000,:]

countFeature = len(X_trainAll[0,:])-1

X_train = X_train1[:,0:countFeature]
Y_train = X_train1[:,countFeature]
Y_train = to_categorical(Y_train, num_classes=2, dtype='float32')

X_test1 = X_trainAll[7000:,:]
X_test = X_test1[:,0:countFeature]
ans  = X_test1[:,countFeature]


#
model = Sequential()
model.add(Dense(20000, input_dim=40, activation="relu"))
model.add(Dense(1000, activation="relu"))
model.add(Dense(2, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size = 200, epochs = 5)

# model = load_model('fashion_mnist_dense.h5')

scores = model.evaluate(X_train, Y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

pred = model.predict(X_test)

ones = 0
zeros =0
for i in range(len(pred)):
    if (np.argmax(pred[i])==1):
        ones = ones + 1
    elif (np.argmax(pred[i])==0):
         zeros = zeros + 1

print("=====================")
print("Здоровых = ", zeros)
print("Больных = ", ones )

# for i in range(len(pred)):
    # print("Класс: ", np.argmax(pred[i]),"  вероятность: " , pred[i][ np.argmax(pred[i])])


ans = to_categorical(ans, num_classes=2, dtype='float32')
scores = model.evaluate(X_test, ans)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))