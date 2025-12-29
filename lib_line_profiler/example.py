import line_profiler


profile = line_profiler.LineProfiler()

@profile
def run():
    for i in range(10000):
        print('Hello World')


if __name__ == '__main__':
    run()
    profile.print_stats()
