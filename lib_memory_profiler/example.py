from memory_profiler import profile


EXAMPLE_OBJECT = {
    'a': 'x',
    'b': 'y',
    'c': 'z',
}


def _example_function() -> dict:
    return {v:k for k,v in EXAMPLE_OBJECT.items()}


EXAMPLE_FUNCTION = _example_function

@profile
def run():
    return EXAMPLE_FUNCTION()


if __name__ == '__main__':
    run()
