import tensorflow as tf
import numpy as np


# Network parameters
TOTAL_WORDS = len(['abc', 'def'])

N_HIDDEN_1 = 10        # Number of features in the first layer
N_HIDDEN_2 = 5         # Number of features in the second layer
N_INPUT = TOTAL_WORDS  # Words in vocabulary
N_CLASSES = 3          # Categories: graphics, sci.space and baseball


def create_multilayer_perceptron(n_input, n_hidden_1, n_hidden_2, n_classes):
    """Create a multilayer perceptron model using Keras Sequential API."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(n_input,)),
        tf.keras.layers.Dense(n_hidden_1, activation='relu', name='hidden_1'),
        tf.keras.layers.Dense(n_hidden_2, activation='relu', name='hidden_2'),
        tf.keras.layers.Dense(n_classes, activation=None, name='output')  # Linear activation for logits
    ])
    return model


def run():
    # Create the model
    model = create_multilayer_perceptron(N_INPUT, N_HIDDEN_1, N_HIDDEN_2, N_CLASSES)

    # Compile the model with loss and optimizer
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Print model summary
    model.summary()

    # Example: Create fake data for demonstration
    # In real usage, replace it with your actual training data
    dummy_input = np.random.rand(10, N_INPUT).astype(np.float32)
    dummy_labels = np.eye(N_CLASSES)[np.random.choice(N_CLASSES, 10)]  # One-hot encoded labels

    # Train the model (example with fake data)
    model.fit(dummy_input, dummy_labels, epochs=5, verbose=1)

    # Make predictions
    predictions = model.predict(dummy_input[:2])
    print("\nPredictions (logits):", predictions)
    print("Predicted classes:", np.argmax(predictions, axis=1))

    return model


if __name__ == '__main__':
    result_model = run()