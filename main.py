import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add, LayerNormalization, Attention, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np

# Data generation 
np.random.seed(42)
t = np.linspace(0, 10, 1000)
X = np.random.rand(1000, 10)
for i in range(10):
    X[:, i] += np.sin(2 * np.pi * i * t)
y = X.sum(axis=1) + np.random.normal(0, 0.1, 1000)

# Add polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True)
X = poly.fit_transform(X)

# Split and standardize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Fourier Transform layer
class FFTLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        fft = tf.signal.fft(tf.cast(inputs, tf.complex64))
        components = [tf.math.real(fft), tf.math.imag(fft), tf.abs(fft), tf.math.angle(fft)]
        return tf.concat(components, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * 4)

# Updated Hilbert Transform layer
class HilbertLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # Compute FFT of the input
        fft = tf.signal.fft(tf.cast(inputs, tf.complex64))
        
        # Create a frequency filter
        N = tf.shape(inputs)[-1]
        half = tf.cast(tf.math.ceil(tf.cast(N, tf.float32) / 2.0), tf.int32)
        filter_ones = tf.ones([half], dtype=tf.complex64)
        filter_zeros = tf.zeros([N - half], dtype=tf.complex64)
        h = tf.concat([filter_ones, filter_zeros], axis=0) * 2.0
        h = tf.concat([[1.0], h[1:half], [1.0], h[half+1:]], axis=0)

        # Apply the filter
        hilbert = tf.signal.ifft(fft * h)

        components = [tf.math.real(hilbert), tf.math.imag(hilbert),
                      tf.abs(hilbert), tf.math.angle(hilbert)]
        return tf.concat(components, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * 4)

# Residual Block
def residual_block(x, units):
    shortcut = Dense(units, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(units, activation=None)(x)
    x = Add()([x, shortcut]) 
    x = LayerNormalization()(x)
    return x

# Model with Fourier, Hilbert, and Deep Feature Extraction
def create_model(input_shape):
    inputs = Input(shape=(input_shape,))

    # Fourier Transform
    fft = FFTLayer()(inputs)

    # Hilbert Transform
    hilbert = HilbertLayer()(inputs)

    # Combine all features
    all_features = tf.keras.layers.concatenate([inputs, fft, hilbert])

    # Reshape for attention layer
    reshaped_features = Reshape((1, all_features.shape[1]))(all_features)

    # Dense layers with residual connections and attention mechanism
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(reshaped_features)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = residual_block(x, 256)
    x = Attention()([x, x])
    x = residual_block(x, 128)
    x = Dropout(0.3)(x)
    x = Reshape((-1,))(x)  # Flatten the output before the final dense layer
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile the model
model = create_model(input_shape=X_train.shape[1])
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Implement callbacks
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)
early_stopping = EarlyStopping(patience=20, monitor='val_loss', restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2,
                    callbacks=[tensorboard, early_stopping, model_checkpoint, reduce_lr])

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)

# Inverse transform predictions and y_test to original scale
predictions = model.predict(X_test)
predictions = scaler_y.inverse_transform(predictions)
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

print(f'Test MAE: {mae}')
print(f'Test Loss: {loss}')
print(predictions[:5])
