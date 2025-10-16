import time

def throttle_example(func, delay):
    """
    Throttle the execution of a function so that it can only be called
    once within the given delay period.

    :param func: Function to throttle
    :param delay: Delay time in seconds
    """
    last_called: list = [0]  # Use a list to store mutable last_called time

    def wrapper(*args, **kwargs):
        nonlocal last_called
        current_time = time.time()
        if current_time - last_called[0] >= delay:
            last_called[0] = current_time
            return func(*args, **kwargs)
        else:
            print("Throttled: Function call ignored")

    return wrapper

# Example Usage
if __name__ == "__main__":
    def example_task():
        print("Task executed")

    throttled_task = throttle_example(example_task, 2)

    # Call the throttled task multiple times
    for _ in range(5):
        throttled_task()
        time.sleep(1)  # Simulate calling with 1 second interval
