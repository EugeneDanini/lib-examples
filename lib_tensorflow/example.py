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
    """A neural network text classifier using TensorFlow."""

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

        # TensorFlow components
        self.input_tensor = None
        self.output_tensor = None
        self.prediction = None
        self.loss = None
        self.optimizer = None
        self.saver = None

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
        vector = np.zeros(self.total_words, dtype=float)
        for word in text.split(' '):
            word_lower = word.lower()
            if word_lower in self.word2index:
                vector[self.word2index[word_lower]] += 1
        return vector

    def category_to_vector(self, category):
        """Convert category index to one-hot encoded vector."""
        y = np.zeros(self.n_classes, dtype=float)
        y[category] = 1.0
        return y

    def get_batch(self, data, i, batch_size):
        """Get a batch of training data."""
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        texts = data.data[start_idx:end_idx]
        categories = data.target[start_idx:end_idx]

        batch_vectors = [self.text_to_vector(text) for text in texts]
        batch_labels = [self.category_to_vector(cat) for cat in categories]

        return np.array(batch_vectors), np.array(batch_labels)

    def multilayer_perceptron(self, input_tensor, weights, biases):
        """Build a multilayer perceptron model."""
        # First hidden layer with RELU activation
        layer_1 = tf.nn.relu(tf.add(tf.matmul(input_tensor, weights['h1']), biases['b1']))

        # Second hidden layer with RELU activation
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

        # Output layer
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

        return out_layer

    def build_model(self):
        """Build the neural network model."""
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.disable_v2_behavior()

        # Define placeholders
        self.input_tensor = tf.compat.v1.placeholder(
            tf.float32, [None, self.total_words], name="input"
        )
        self.output_tensor = tf.compat.v1.placeholder(
            tf.float32, [None, self.n_classes], name="output"
        )

        # Define weights and biases
        weights = {
            'h1': tf.Variable(tf.random.normal([self.total_words, self.n_hidden_1])),
            'h2': tf.Variable(tf.random.normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random.normal([self.n_hidden_2, self.n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random.normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random.normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random.normal([self.n_classes]))
        }

        # Build model
        self.prediction = self.multilayer_perceptron(self.input_tensor, weights, biases)

        # Define loss and optimizer
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.prediction, labels=self.output_tensor
            )
        )
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.learning_rate
        ).minimize(self.loss)

        # Initialize saver
        self.saver = tf.compat.v1.train.Saver()

        logging.info("Model built successfully")

    def train(self, train_data, test_data, model_path="/tmp/model.ckpt"):
        """Train the model."""
        init = tf.compat.v1.global_variables_initializer()

        with tf.compat.v1.Session() as sess:
            sess.run(init)

            # Training cycle
            total_batch = int(len(train_data.data) / self.batch_size)

            for epoch in range(self.training_epochs):
                avg_cost = 0.0

                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = self.get_batch(train_data, i, self.batch_size)

                    # Run optimization
                    c, _ = sess.run(
                        [self.loss, self.optimizer],
                        feed_dict={self.input_tensor: batch_x, self.output_tensor: batch_y}
                    )
                    avg_cost += c / total_batch

                # Log progress
                logging.info("Epoch: %04d loss=%.9f", epoch + 1, avg_cost)

            logging.info("Optimization Finished!")

            # Evaluate on test data
            correct_prediction = tf.equal(
                tf.argmax(self.prediction, 1),
                tf.argmax(self.output_tensor, 1)
            )
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            total_test_data = len(test_data.target)
            batch_x_test, batch_y_test = self.get_batch(test_data, 0, total_test_data)

            test_accuracy = accuracy.eval({
                self.input_tensor: batch_x_test,
                self.output_tensor: batch_y_test
            })
            logging.info("Test Accuracy: %.4f", test_accuracy)

            # Save the model
            save_path = self.saver.save(sess, model_path)
            logging.info("Model saved in path: %s", save_path)

    def predict(self, texts, model_path="/tmp/model.ckpt"):
        """Predict categories for given texts."""
        with tf.compat.v1.Session() as sess:
            self.saver.restore(sess, model_path)
            logging.info("Model restored.")

            # Convert texts to vectors
            if isinstance(texts, str):
                texts = [texts]

            input_vectors = np.array([self.text_to_vector(text) for text in texts])

            # Make prediction
            classifications = sess.run(
                tf.argmax(self.prediction, 1),
                feed_dict={self.input_tensor: input_vectors}
            )

            return classifications


def demo_basic_tensorflow():
    """Demonstrate basic TensorFlow operations."""
    logging.info("=== Basic TensorFlow Demo ===")

    my_graph = tf.Graph()
    with tf.compat.v1.Session(graph=my_graph) as sess:
        x = tf.constant([1, 3, 6])
        y = tf.constant([1, 1, 1])
        op = tf.add(x, y)
        result = sess.run(fetches=op)
        logging.info("Result: %s", result)


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


def run():
    """Main function to run the text classification example."""
    # Run basic demos
    demo_basic_tensorflow()
    demo_text_vectorization()

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

    # Demo batch creation
    batch_x, batch_y = classifier.get_batch(newsgroups_train, 1, 100)
    logging.info("Batch shape - texts: %s, labels: %s", batch_x.shape, batch_y.shape)

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

    # Multiple predictions
    test_texts = newsgroups_test.data[:10]
    predictions = classifier.predict(test_texts)
    correct_categories = newsgroups_test.target[:10]

    logging.info("Predicted categories (10 samples): %s", predictions)
    logging.info("Correct categories (10 samples): %s", correct_categories)


if __name__ == '__main__':
    run()
