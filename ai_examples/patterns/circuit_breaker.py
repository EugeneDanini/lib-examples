class CircuitBreaker:
    def __init__(self, failure_threshold, recovery_timeout):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
        self.state = "CLOSED"  # Possible states: CLOSED, OPEN

    def call(self, func, *args, **kwargs):
        # Handle circuit logic before allowing service calls
        if self.is_open():
            raise Exception("Circuit is open. Service temporarily unavailable.")
        
        try:
            result = func(*args, **kwargs)
            self.reset()
            return result
        except Exception:
            self.fail()
            raise

    def is_open(self):
        # Check if the circuit is open and decides it's ready to reset
        if self.state == "OPEN":
            import time
            if (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.reset()
            else:
                return True
        return False

    def reset(self):
        # Reset the circuit breaker state
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = None

    def fail(self):
        # Record a failure and open circuit if threshold reached
        import time
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
