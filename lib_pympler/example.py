from pympler import asizeof, classtracker, tracker

EXAMPLE_OBJECT = {
    'a': 'x',
    'b': 'y',
    'c': 'z',
}


def run():
    _classtracker = classtracker.ClassTracker()
    _classtracker.track_class(tracker.SummaryTracker)
    _classtracker.create_snapshot()
    _tracker = tracker.SummaryTracker()
    print(asizeof.asized(EXAMPLE_OBJECT, detail=1).format())
    _tracker.print_diff()
    _classtracker.create_snapshot()
    _classtracker.stats.print_summary()


if __name__ == '__main__':
    run()