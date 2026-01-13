import logging
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class TextClassifier:
    """A neural network text classifier using TensorFlow v2 and Keras."""

    def __init__(
        self,
        n_classes=3,
        n_hidden_1=100,
        n_hidden_2=100,
        learning_rate=0.01,
        training_epochs=10,
        batch_size=150,
    ):
        self.n_classes = n_classes
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size

        self.vocab = None
        self.word2index = None
        self.total_words = 0

        # TensorFlow/Keras model
        self.model = None

    def build_vocab(self, texts):
        """Build vocabulary from texts."""
        vocab = Counter()
        for text in texts:
            for word in text.split(' '):
                vocab[word.lower()] += 1

        self.vocab = vocab
        self.total_words = len(vocab)
        self.word2index = {word: i for i, word in enumerate(vocab)}

        logging.info("Total words in vocabulary: %d", self.total_words)

    def text_to_vector(self, text):
        """Convert text to vector representation."""
        vector = np.zeros(self.total_words, dtype=np.float32)
        for word in text.split(' '):
            word_lower = word.lower()
            if word_lower in self.word2index:
                vector[self.word2index[word_lower]] += 1
        return vector

    def texts_to_vectors(self, texts):
        """Convert multiple texts to vectors."""
        if isinstance(texts, str):
            texts = [texts]
        return np.array([self.text_to_vector(text) for text in texts])

    def prepare_dataset(self, data):
        """Convert dataset to tensors."""
        x = self.texts_to_vectors(data.data)
        # Convert to one-hot encoding
        y = tf.keras.utils.to_categorical(data.target, num_classes=self.n_classes)
        return x, y

    def build_model(self):
        """Build the neural network model using Keras Sequential API."""
        self.model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.InputLayer(input_shape=(self.total_words,)),

            # First hidden layer with RELU activation
            tf.keras.layers.Dense(
                self.n_hidden_1,
                activation='relu',
                kernel_initializer='random_normal',
                bias_initializer='random_normal'
            ),

            # Second hidden layer with RELU activation
            tf.keras.layers.Dense(
                self.n_hidden_2,
                activation='relu',
                kernel_initializer='random_normal',
                bias_initializer='random_normal'
            ),

            # Output layer with softmax activation
            tf.keras.layers.Dense(
                self.n_classes,
                kernel_initializer='random_normal',
                bias_initializer='random_normal'
            )
        ])

        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        logging.info("Model built successfully")
        self.model.summary(print_fn=logging.info)

    def train(self, train_data, test_data, model_path="/tmp/model.keras"):
        """Train the model."""
        # Prepare datasets
        x_train, y_train = self.prepare_dataset(train_data)
        x_test, y_test = self.prepare_dataset(test_data)

        logging.info("Training dataset shape: X=%s, y=%s", x_train.shape, y_train.shape)
        logging.info("Test dataset shape: X=%s, y=%s", x_test.shape, y_test.shape)

        # Create a custom callback for logging
        class LoggingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logging.info(
                    "Epoch: %04d loss=%.9f accuracy=%.4f val_loss=%.9f val_accuracy=%.4f",
                    epoch + 1,
                    logs.get('loss', 0),
                    logs.get('accuracy', 0),
                    logs.get('val_loss', 0),
                    logs.get('val_accuracy', 0)
                )

        # Train the model
        history = self.model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.training_epochs,
            validation_data=(x_test, y_test),
            callbacks=[LoggingCallback()],
            verbose=0  # We use our custom callback for logging
        )

        logging.info("Optimization Finished!")

        # Evaluate on test data
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        logging.info("Test Accuracy: %.4f", test_accuracy)
        logging.info("Test Loss: %.4f", test_loss)

        # Save the model
        self.model.save(model_path)
        logging.info("Model saved in path: %s", model_path)

        return history

    def predict(self, texts, model_path=None):
        """Predict categories for given texts."""
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
            logging.info("Model restored from: %s", model_path)

        # Convert texts to vectors
        input_vectors = self.texts_to_vectors(texts)

        # Make prediction (get logits)
        logits = self.model.predict(input_vectors, verbose=0)

        # Convert to class predictions
        predictions = tf.argmax(logits, axis=1).numpy()

        return predictions

    def predict_proba(self, texts, model_path=None):
        """Predict class probabilities for given texts."""
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
            logging.info("Model restored from: %s", model_path)

        # Convert texts to vectors
        input_vectors = self.texts_to_vectors(texts)

        # Make prediction (get logits)
        logits = self.model.predict(input_vectors, verbose=0)

        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(logits).numpy()

        return probabilities


def demo_basic_tensorflow():
    """Demonstrate basic TensorFlow v2 operations."""
    logging.info("=== Basic TensorFlow v2 Demo ===")

    # TensorFlow v2 uses eager execution by default - no sessions needed
    x = tf.constant([1, 3, 6])
    y = tf.constant([1, 1, 1])
    result = tf.add(x, y)

    logging.info("Result: %s", result.numpy())

    # Demonstrate tensor operations
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])
    matmul_result = tf.matmul(a, b)

    logging.info("Matrix multiplication result:\n%s", matmul_result.numpy())


def demo_text_vectorization():
    """Demonstrate text vectorization."""
    logging.info("=== Text Vectorization Demo ===")

    vocabulary = Counter()
    text = "Hi from Brazil"

    for word in text.split(' '):
        vocabulary[word.lower()] += 1

    word2index = {word: i for i, word in enumerate(vocabulary)}
    total_words = len(vocabulary)

    # Vectorize text
    matrix = np.zeros(total_words, dtype=float)
    for word in text.split():
        matrix[word2index[word.lower()]] += 1

    logging.info("Text '%s' vectorized: %s", text, matrix)

    # Vectorize a single word
    matrix = np.zeros(total_words, dtype=float)
    single_word = "Hi"
    for word in single_word.split():
        matrix[word2index[word.lower()]] += 1

    logging.info("Text '%s' vectorized: %s", single_word, matrix)


def demo_keras_sequential():
    """Demonstrate Keras Sequential API."""
    logging.info("=== Keras Sequential API Demo ===")

    # Create a simple sequential model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Generate dummy data
    x_dummy = np.random.random((100, 10))
    y_dummy = tf.keras.utils.to_categorical(np.random.randint(3, size=(100, 1)), num_classes=3)

    # Train for a few epochs
    logging.info("Training simple model...")
    model.fit(x_dummy, y_dummy, epochs=3, batch_size=32, verbose=0)

    # Make predictions
    predictions = model.predict(x_dummy[:5], verbose=0)
    logging.info("Sample predictions shape: %s", predictions.shape)


def run():
    """Main function to run the text classification example."""
    # Run basic demos
    demo_basic_tensorflow()
    demo_text_vectorization()
    demo_keras_sequential()

    # Text classification with neural network
    logging.info("=== Text Classification with Neural Network ===")

    # Load dataset
    categories = ["comp.graphics", "sci.space", "rec.sport.baseball"]
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

    logging.info("Total texts in train: %d", len(newsgroups_train.data))
    logging.info("Total texts in test: %d", len(newsgroups_test.data))
    logging.info("Sample text: %s", newsgroups_train.data[0][:100] + "...")
    logging.info("Sample category: %d", newsgroups_train.target[0])

    # Initialize classifier
    classifier = TextClassifier(
        n_classes=3,
        n_hidden_1=100,
        n_hidden_2=100,
        learning_rate=0.01,
        training_epochs=10,
        batch_size=150
    )

    # Build vocabulary from train and test data
    all_texts = list(newsgroups_train.data) + list(newsgroups_test.data)
    classifier.build_vocab(all_texts)

    # Check word index
    if 'the' in classifier.word2index:
        logging.info("Index of the word 'the': %d", classifier.word2index['the'])

    # Build and train model
    classifier.build_model()
    classifier.train(newsgroups_train, newsgroups_test)

    # Make predictions
    logging.info("=== Making Predictions ===")

    # Single prediction
    test_text = newsgroups_test.data[5]
    logging.info("Test text: %s", test_text[:100] + "...")
    logging.info("Correct category: %d", newsgroups_test.target[5])

    prediction = classifier.predict(test_text)
    logging.info("Predicted category: %s", prediction)

    # Predict with probabilities
    probabilities = classifier.predict_proba(test_text)
    logging.info("Prediction probabilities: %s", probabilities)

    # Multiple predictions
    test_texts = newsgroups_test.data[:10]
    predictions = classifier.predict(test_texts)
    correct_categories = newsgroups_test.target[:10]

    logging.info("Predicted categories (10 samples): %s", predictions)
    logging.info("Correct categories (10 samples): %s", correct_categories)

    # Calculate accuracy
    accuracy = np.sum(predictions == correct_categories) / len(correct_categories)
    logging.info("Sample accuracy (10 samples): %.2f%%", accuracy * 100)


if __name__ == '__main__':
    run()
