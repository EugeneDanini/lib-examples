import objgraph

EXAMPLE_OBJECT = {
    'a': 'x',
    'b': 'y',
    'c': 'z',
}


def run():
    objgraph.show_refs(EXAMPLE_OBJECT, filename='lib_objgraph/example.png')
    objgraph.show_most_common_types()


if __name__ == '__main__':
    run()
