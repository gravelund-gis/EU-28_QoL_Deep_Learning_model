"""
Deep Learning Keras/TensorFlow regression model of EU-28 Quality of Life (QoL) levels for year 2014,
using 11 socioeconomic indicators, retrieved from Eurostat, as predictors and 1 climatic comfort predictor.
The target is 2013 EUSILC QoL survey data (ground truth).

A model to quantitatively estimate the spatial well-being (QoL) distribution across the EU-28 member states down
to the municipal level. The model is weight-driven and based on Eurostat statistics on objective key QoL indicators,
identified by the 2016 Eurostat Analytical report on subjective well-being and the 2013 EU-SILC ad-hoc module on
well-being. Additionally, some Europe 2020 strategy targets of the European Commission, deemed to be important to a
sense of personal well-being, are included, such as the risk of poverty or social exclusion and advanced educational
attainment.

A climatic comfort component based on 1961-1990 climatic normals is added to estimate the importance of (a static)
climate to QoL. Thermal comfort levels are obtained using the Universal Thermal Climate Index (UTCI) and
Predicted Mean Vote (PMV), and overall climatic comfort levels are obtained as a weighted linear combination
based on the classical Tourism Climatic Index (TCI).

To evaluate the performance of the Deep Learning model, it is compared to the results from the 2013 EU subjective
well-being survey (the ground truth).
"""
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Import raw data as pandas dataframe "df", and instantiate a scaler
df = pd.read_csv('merged_raw_data_2014.csv', delimiter=',')
scaler = StandardScaler()


# Pearson r correlation coefficient function
def pearson_r(array1, array2):
    """
    Compute Pearson correlation coefficient between two arrays.
    """
    corr_mat = np.corrcoef(array1, array2)
    # Return entry [0,1]
    return corr_mat[0, 1]


# Create and scale the target and predictor arrays
target = df.EUSILC.as_matrix()
target_scaled = scaler.fit_transform(target)
predictors = df.drop(
	['EUSILC', 'WS_QoL', 'WS_QoLadj','CNTR_CODE', 'OBJECTID', 'POINTID', 'NUTS_ID', 'STAT_LEVL_', 'LAT', 'LON'],
	axis=1).as_matrix()

predictors_scaled = scaler.fit_transform(predictors)
n_cols = predictors_scaled.shape[1]
layers = 3

# Create the model and add layers
model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(n_cols,)))
for i in range(1, layers):
    model.add(Dense(50, activation='relu'))

# Add final output layer
model.add(Dense(1))

# Compile the model, with mean squared error loss function as a measure of model success
model.compile(optimizer='adam', loss='mean_squared_error')

# Provide early stopping if no improvements are shown
early_stopping_monitor = EarlyStopping(patience=7)

# Fit the model, splitting into training and hold-out sets
history = model.fit(predictors_scaled, target_scaled,
                    validation_split=0.3, epochs=50, callbacks=[early_stopping_monitor])

print("Min. validation loss: ", np.min(history.history['val_loss']))

# MODEL PREDICTING
#--------------------
# Calculate predictions and flatten the array. The predictions will be probabilities.
predictions = model.predict(predictors_scaled)
predictions_flat = predictions.ravel()

# Shift the predictions up by the difference in means to the standard 0-10 scale of the EUSILC survey data
predictions_scaled = predictions_flat - np.mean(predictions_flat) + np.mean(target)

# Create combined dataframe
probs = [('POINTID', df['POINTID']),
         ('QoL_pred_prob', list(predictions_scaled))]

df_predicted = pd.DataFrame.from_items(probs)

df_combined = pd.merge(df, df_predicted, how='left', on='POINTID')
print(df_combined.head())

# Create plotting variables
x = df_combined['EUSILC']
y = df_combined['QoL_pred_prob']
y_thesis = df_combined['WS_QoLadj']

# Calculate Pearson R
pearson_R = pearson_r(x, y)
print('Deep Learning R = ', pearson_R)

# Save the combined model output data to disk, appending the unique pearson R value
df_combined.to_csv(r'E:\PYTHON\DataCamp\Courses\Deep_Learning\QoL_results\QoL_pred_2014_{}.csv'.format(pearson_R))

# PLOTTING
#--------------------
fig = plt.figure()
fig.suptitle('Machine Learning model prediction of year 2014 EU-28 Quality of Life (QoL) levels (standard scale: 0-10)')

ax = plt.subplot(2,2,1)
ax.set_title('Deep Learning model weights optimization')
ax.plot(history.history['val_loss'], 'r')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')

ax = plt.subplot(2,2,2)
ax.set_title('Regression plot of original thesis QoL result (blue),\n\
            Machine Learning QoL model prediction (red)\n\
            and 2013 EU-SILC survey (ground truth)')
ax.plot(x, y, 'red', marker='.', linestyle='none', ms=3, alpha=.2, label='QoL Deep Learning model')
ax.plot(x, y_thesis, 'blue', marker='.', linestyle='none', ms=3, alpha=.2, label='Thesis QoL model result')
ax.margins(0.02)
plt.annotate('R = ' + str(round(pearson_R, 2)), xy=(np.max(x), np.max(y) - 4), fontsize=9)
plt.xlabel('EU-SILC ground truth')
plt.ylabel('QoL level')
ax.legend(loc='upper left')
# Perform a linear regression, to find the slope a and intercept b, using np.polyfit() of order 1: a, b
a, b = np.polyfit(x, y, 1)
# Make theoretical line to plot using the found a, b
X = np.array([0, 10])
y_pred = a * X + b
# Add regression line to the plot
ax.plot(X, y_pred)

ax = plt.subplot(2,2,3)
# Plot a linear regression
ax.set_title('Residuals of Machine Learning QoL model prediction\n and 2013 EU-SILC survey (ground truth)')
sns.residplot(x='EUSILC', y='QoL_pred_prob', data=df_combined, color='red',
              scatter_kws={"s": 5, "alpha": .6})

# Draw the plot
plt.tight_layout()
plt.show()
