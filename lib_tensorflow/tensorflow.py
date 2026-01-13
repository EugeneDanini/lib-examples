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

my_graph = tf.Graph()
with tf.compat.v1.Session(graph=my_graph) as sess:
    x = tf.constant([1,3,6])
    y = tf.constant([1,1,1])
    op = tf.add(x,y)
    result = sess.run(fetches=op)
    logging.info("Result: %s", result)

vocab = Counter()

text = "Hi from Brazil"

for word in text.split(' '):
    word_lowercase = word.lower()
    vocab[word_lowercase] += 1


def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i

    return word2index


word2index = get_word_2_index(vocab)

total_words = len(vocab)
matrix = np.zeros((total_words), dtype=float)

for word in text.split():
    matrix[word2index[word.lower()]] += 1

logging.info("Hi from Brazil: %s", matrix)

matrix = np.zeros((total_words), dtype=float)
text = "Hi"
for word in text.split():
    matrix[word2index[word.lower()]] += 1

logging.info("Hi: %s", matrix)

categories = ["comp.graphics","sci.space","rec.sport.baseball"]
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)


logging.info("Total texts in train: %d", len(newsgroups_train.data))
logging.info("Total texts in test: %d", len(newsgroups_test.data))

logging.info("Text: %s", newsgroups_train.data[0])
logging.info("Category: %d", newsgroups_train.target[0])

vocab = Counter()

for text in newsgroups_train.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

for text in newsgroups_test.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

logging.info("Total words: %d", len(vocab))

total_words = len(vocab)


def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i

    return word2index


word2index = get_word_2_index(vocab)

logging.info("Index of the word 'the': %d", word2index['the'])


def text_to_vector(text):
    layer = np.zeros(total_words, dtype=float)
    for word in text.split(' '):
        layer[word2index[word.lower()]] += 1

    return layer


def category_to_vector(category):
    y = np.zeros((3), dtype=float)
    if category == 0:
        y[0] = 1.
    elif category == 1:
        y[1] = 1.
    else:
        y[2] = 1.

    return y


def get_batch(df, i, batch_size):
    batches = []
    results = []
    texts = df.data[i * batch_size:i * batch_size + batch_size]
    categories = df.target[i * batch_size:i * batch_size + batch_size]

    for text in texts:
        layer = text_to_vector(text)
        batches.append(layer)

    for category in categories:
        y = category_to_vector(category)
        results.append(y)

    return np.array(batches), np.array(results)

logging.info("Each batch has 100 texts and each matrix has 119930 elements (words): %s", get_batch(newsgroups_train,1,100)[0].shape)

logging.info("Each batch has 100 labels and each matrix has 3 elements (3 categories): %s", get_batch(newsgroups_train,1,100)[1].shape)

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 150
display_step = 1

# Network Parameters
n_hidden_1 = 100      # 1st layer number of features
n_hidden_2 = 100       # 2nd layer number of features
n_input = total_words # Words in vocab
n_classes = 3         # Categories: graphics, sci.space and baseball

tf.compat.v1.disable_eager_execution()
input_tensor = tf.compat.v1.placeholder(tf.float32,[None, n_input],name="input")
output_tensor = tf.compat.v1.placeholder(tf.float32,[None, n_classes],name="output")


def multilayer_perceptron(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1 = tf.nn.relu(layer_1_addition)

    # Hidden layer with RELU activation
    layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2 = tf.nn.relu(layer_2_addition)

    # Output layer
    out_layer_multiplication = tf.matmul(layer_2, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']

    return out_layer_addition

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random.normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random.normal([n_hidden_1])),
    'b2': tf.Variable(tf.random.normal([n_hidden_2])),
    'out': tf.Variable(tf.random.normal([n_classes]))
}

# Construct model
prediction = multilayer_perceptron(input_tensor, weights, biases)

# Define loss and optimizer
tf.compat.v1.disable_v2_behavior()
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# Initializing the variables
# init = tf.global_variables_initializer()

# [NEW] Add ops to save and restore all the variables
tf.compat.v1.disable_eager_execution()
init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(newsgroups_train.data) / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = get_batch(newsgroups_train, i, batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            c, _ = sess.run([loss, optimizer], feed_dict={input_tensor: batch_x, output_tensor: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            logging.info("Epoch: %04d loss=%.9f", epoch + 1, avg_cost)
    logging.info("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    total_test_data = len(newsgroups_test.target)
    batch_x_test, batch_y_test = get_batch(newsgroups_test, 0, total_test_data)
    logging.info("Accuracy: %s", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))

    # [NEW] Save the variables to disk
    save_path = saver.save(sess, "/tmp/model.ckpt")
    logging.info("Model saved in path: %s", save_path)

text_for_prediction = newsgroups_test.data[5]

logging.info("Text: %s", text_for_prediction)

logging.info("Text correct category: %d", newsgroups_test.target[5])

# Convert text to vector so we can send it to our model
vector_txt = text_to_vector(text)
# Wrap vector like we do in get_batches()
input_array = np.array([vector_txt])

tf.compat.v1.disable_eager_execution()
saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    saver.restore(sess, "/tmp/model.ckpt")
    logging.info("Model restored.")

    classification = sess.run(tf.argmax(prediction, 1), feed_dict={input_tensor: input_array})
    logging.info("Predicted category: %s", classification)

# Get 10 texts to make a prediction

x_10_texts, y_10_correct_labels = get_batch(newsgroups_test, 0, 10)

tf.compat.v1.disable_eager_execution()
saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    saver.restore(sess, "/tmp/model.ckpt")
    logging.info("Model restored.")

    classification = sess.run(tf.argmax(prediction, 1), feed_dict={input_tensor: x_10_texts})
    logging.info("Predicted categories: %s", classification)


logging.info("Correct categories: %s", np.argmax(y_10_correct_labels, 1))