# Databricks notebook source
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

# COMMAND ----------

# Plot time-series graph. time=x, series=y
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

# Add a slope to the series
def trend(time, slope=0):
    return slope * time

# Create a seasonal pattern
def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

# Add seasonality (cosine curve) with given amplitude and phase
def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

# Add random noise
def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

# Create a time range for 4 years of daily
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)  
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

# Split into a training (first 1000 days) and validation (everything else) dataset
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# We will use this to create a windowed dataset
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

# COMMAND ----------

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
# Create a tensorflow dataset from our seriese
# [1,2,3,4,5,6,7,8,9,10]
  dataset = tf.data.Dataset.from_tensor_slices(series)
  
# Create a windowed dataset: window_size = # of elements per window, shift = how much to shift by each window, drop_remainder = only show complete windows
# [1,2,3,4],[2,3,4,5],[3,4,5,6],...
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  
# Split into a dataset with everything except last value as features, and last item as the label (ie. first 4 elements to predict 5th element in each window)
# Shuffle the data with a specified shuffle size
# [[1,2,3],[4]],[[2,3,4],[5]],[[3,4,5],[6]],...
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  
# Batch the data into groups of batch_size for training (ie. 32 feature windows and their corresponding labels)
# [[1,2,3],[2,3,4]][[4],[5]],...
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

# COMMAND ----------

# Create a windowed dataset our of the data
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

# Create a linear DNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"), # start with a shape = window_size,
    tf.keras.layers.Dense(10, activation="relu"), # Add a layer
    tf.keras.layers.Dense(1) # End with a single value for a given window
])

# Compile using SGD to minimize MSE and fit the data
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
model.fit(dataset,epochs=100,verbose=0)

# COMMAND ----------

forecast = []

# Build a forecast for every window (ie. pass in a window of preceding values in series)
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

# We only care about how we perform after the split
forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]

# Plot our actual (x_valid) vs. predictions (results)
f = plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)

display(f)

# COMMAND ----------

# Compute the MAE
tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()

# COMMAND ----------

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(10, activation="relu"), 
    tf.keras.layers.Dense(1)
])

# Configure a callback that adjusts the learning rate at each epoch, so we can figure out the optimal learning rate
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=100, callbacks=[lr_schedule], verbose=0)

# COMMAND ----------

# Compare the loss per epoch to the learning rate at each epoch
lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])
f = plt.show()
display(f)

# COMMAND ----------

# Let's retrain using a learning rate of 8x10^-6
window_size = 30
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1)
])

# Increase to 500 epochs
optimizer = tf.keras.optimizers.SGD(lr=8e-6, momentum=0.9)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=500, verbose=0)

# COMMAND ----------

loss = history.history['loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'b', label='Training Loss')
display(plt.show())

# COMMAND ----------

# Plot all but the first 10 - note, even at 500 epochs, the model is learning well
loss = history.history['loss']
epochs = range(10, len(loss))
plot_loss = loss[10:]
print(plot_loss)
plt.plot(epochs, plot_loss, 'b', label='Training Loss')
display(plt.show())

# COMMAND ----------

forecast = []
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


f = plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)

display(f)

# COMMAND ----------

tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()

# COMMAND ----------

