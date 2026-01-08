import tensorflow as tf

def run():
    graph = tf.Graph()

    with tf.compat.v1.Session(graph=graph) as session:
        x = tf.constant([1, 3, 6])
        y = tf.constant([1, 0, 1])
        operation = tf.add(x, y)
        return session.run(fetches=operation)


if __name__ == '__main__':
    print(run())
