import plotly.graph_objects as go
import joblib
import numpy as np

model = joblib.load("model")
predictedAndActual = np.load("predict_actual.npy")

predicted = predictedAndActual[:,:3]
actual = predictedAndActual[:,3:]

#print(predicted.shape, actual.shape)

error = np.abs(actual - predicted)
print(error)

distanceError = []

for i in range(len(error)):
    dist = np.sqrt(error[i][0]**2 + error[i][1]**2 + error[i][2]**2)
    distanceError.append(dist)

print(distanceError)

distanceError = np.array(distanceError)

fig = go.Figure()
fig.add_trace(go.Bar(x=np.arange(len(distanceError)), y=distanceError, name='Distance Error (km)', marker_color='indianred'))
fig.update_layout(
    title='Distance Error between Actual and Predicted Positions using Random Forest Model',
    xaxis_title='Test Sample Index',
    yaxis_title='Distance Error (km)',
)
fig.show()
