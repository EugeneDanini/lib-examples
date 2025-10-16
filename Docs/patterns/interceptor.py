class Interceptor:
    """Defines the interface for interceptors."""
    def intercept(self, request):
        raise NotImplementedError("This method should be overridden in subclasses.")

class LoggingInterceptor(Interceptor):
    """A concrete interceptor for logging request details."""
    def intercept(self, request):
        print(f"Logging request: {request}")
        return request

class RequestProcessor:
    """Handles requests and applies interceptors."""
    def __init__(self):
        self.interceptors = []

    def add_interceptor(self, interceptor):
        self.interceptors.append(interceptor)

    def process_request(self, request):
        for interceptor in self.interceptors:
            request = interceptor.intercept(request)
        return f"Processed request: {request}"
