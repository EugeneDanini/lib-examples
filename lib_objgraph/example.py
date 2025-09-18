import objgraph

from config import BASE_PATH, EXAMPLE_OBJECT


def run():
    objgraph.show_refs(EXAMPLE_OBJECT, filename=f'{BASE_PATH}/example.png')
    objgraph.show_most_common_types()


if __name__ == '__main__':
    run()
