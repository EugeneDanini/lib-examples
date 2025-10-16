# Example demonstrating the Retry Storm AntiPattern

import threading
import time

def retry_logic(client_id, retry_count=3, delay=1):
    """
    Simple retry mechanism illustrating async retries by clients.
    Args:
    - client_id (str): Identifier for the client.
    - retry_count (int): Number of retry attempts. Default is 3.
    - delay (int): Delay between retry attempts. Default is 1 second.
    """
    for attempt in range(1, retry_count + 1):
        print(f"Client {client_id}: Attempt {attempt}")
        time.sleep(delay)

# Simulate multiple clients causing a retry storm
clients = 5  # Number of clients in retry storm

threads = []
for i in range(clients):
    t = threading.Thread(target=retry_logic, args=(f"Client-{i+1}",))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
