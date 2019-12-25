import numpy as np
from keras.models import Sequential
from keras.layers import Dense


def genY():
    y = []
    for i in range(15):
        y.append(1)
    for i in range(179):
        y.append(0)
    return y


path1 = "EEG_Features\\" + str(1) + ".txt"
path2 = "EEG_Features\\test.txt"

X_train = np.loadtxt(path1, delimiter = " ")

X_test = np.loadtxt(path2, delimiter = " ")
Y_train = genY()


model = Sequential()
model.add(Dense(4000, input_dim=40, activation="relu"))
model.add(Dense(2000, activation='relu'))
model.add(Dense(4000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=50, epochs=5)

scores = model.evaluate(X_train, Y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

pred = model.predict_classes(X_test)
print(pred)

