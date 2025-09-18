import objgraph


def run():
    a = 1
    b = {'c': 2}
    objgraph.show_refs([b], filename='example.png')
    objgraph.show_most_common_types()


if __name__ == '__main__':
    run()
