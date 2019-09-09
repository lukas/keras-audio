from preprocess import *
from tensorflow import keras
import glob
import wandb

# Set hyper-parameters
wandb.init()
config = wandb.config
config.max_len = 11
config.buckets = 20


# Cache pre-processed data
if len(glob.glob("*.npy")) == 0:
    save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)

labels = ["bed", "happy", "cat"]

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# # Feature dimension
channels = 1
config.epochs = 50
config.batch_size = 100

num_classes = 3

X_train = X_train.reshape(
    X_train.shape[0], config.buckets, config.max_len, channels)
X_test = X_test.reshape(
    X_test.shape[0], config.buckets, config.max_len, channels)

y_train_hot = keras.utils.to_categorical(y_train)
y_test_hot = keras.utils.to_categorical(y_test)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(
    config.buckets, config.max_len, channels)))
model.add(keras.layers.Dense(num_classes, activation='softmax'))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
config.total_params = model.count_params()


model.fit(X_train[:1000], y_train_hot[:1000], batch_size=config.batch_size, epochs=config.epochs, validation_data=(
    X_test, y_test_hot), callbacks=[wandb.keras.WandbCallback(data_type="image", labels=labels)])
