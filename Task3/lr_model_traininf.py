import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

dataFrame = np.load("dataFrame.npy")
features = dataFrame[:, :-3]
next_pos = dataFrame[:, -3:]

print(features.shape, next_pos.shape)

f_train, f_test, n_train, n_test = train_test_split(features, next_pos, test_size=0.2, random_state=33)

model = LinearRegression()
model.fit(f_train, n_train)

predicted = model.predict(f_test)

model.score(f_test,n_test)

joblib.dump(model, 'lr_model')

lr_predictedAndActual = np.hstack((predicted, n_test))

print(lr_predictedAndActual)

np.save("lr_predict_actual.npy",lr_predictedAndActual)