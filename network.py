import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle


path1 = "EEG_Features\\" + str(1) + ".txt"
path2 = "EEG_Features\\test.txt"

X_train1 = np.loadtxt(path1, delimiter = " ")
X_train1 = shuffle(X_train1)

X_test = np.loadtxt(path2, delimiter = " ")

countFeature = len(X_train1[0,:])-1

X_train = X_train1[:,0:countFeature]

Y_train = X_train1[:,countFeature]


model = Sequential()
model.add(Dense(20000, input_dim=40, activation="relu"))
model.add(Dense(4000, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size = 200, epochs = 1)

scores = model.evaluate(X_train, Y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

pred = model.predict_classes(X_test)

ones = 0
zeros =0
for i in pred:
    if (i == 1):
        ones = ones + 1
    elif i==0:
         zeros = zeros + 1
print("=====================")
print("Здоровых = ", zeros)
print("Больных = ", ones )
# print(pred)

