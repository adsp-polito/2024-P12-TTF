import numpy as np
import tensorflow as tf

epochs=100

es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
)

def make_lstm(input_shape):

    LSTM = tf.keras.models.Sequential()
    LSTM.add(tf.keras.layers.Input(shape=input_shape))
    LSTM.add(tf.keras.layers.LSTM(units=20, return_sequences=True))
    LSTM.add(tf.keras.layers.Dropout(0.5))
    LSTM.add(tf.keras.layers.LSTM(units=10, return_sequences=False))
    LSTM.add(tf.keras.layers.Dropout(0.5))
    LSTM.add(tf.keras.layers.Dense(units=1))
    LSTM.compile(optimizer='adam', loss='mean_squared_error')

    return LSTM
    

def train(X_train: np.ndarray, y_train: np.ndarray, X_val, y_val, path: str):

    model = make_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_data=(X_val, y_val), verbose=1, callbacks = [es_callback])
    
    # Save the model
    model.save(path)


def predict(X_test: np.ndarray, path: str):
    #load the model
    model: tf.keras.models.Sequential = tf.keras.models.load_model(path)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Predict on the test data
    predictions = model.predict(X_test)

    return predictions.reshape(-1)