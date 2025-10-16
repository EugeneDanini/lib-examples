"""
Robustness Patterns in Distributed Systems

Robustness refers to a system's ability to handle and recover from failures,
unexpected inputs, and adverse conditions while maintaining functionality.

Key robustness patterns:
1. Graceful Degradation - Continue operating with reduced functionality
2. Bulkhead Pattern - Isolate failures to prevent cascade
3. Chaos Engineering - Proactively test system resilience
4. Health Checks - Monitor component health
5. Defensive Programming - Validate inputs and handle edge cases
"""

import time
import random
from typing import Any, Callable
from dataclasses import dataclass
from enum import Enum


class ServiceHealth(Enum):
    """Health status of a service"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Result of a health check"""
    status: ServiceHealth
    message: str
    timestamp: float


# 1. GRACEFUL DEGRADATION
class SearchService:
    """
    Search service that degrades gracefully when dependencies fail.
    Falls back to simpler search when advanced features are unavailable.
    """

    def __init__(self):
        self.ml_service_available = True
        self.cache_available = True

    def search(self, query: str) -> list[dict]:
        """Search with graceful degradation"""
        # Try ML-powered search first
        if self.ml_service_available:
            try:
                return self._ml_search(query)
            except Exception as e:
                print(f"ML search failed: {e}. Falling back to basic search.")
                self.ml_service_available = False

        # Fall back to cached results
        if self.cache_available:
            try:
                return self._cached_search(query)
            except Exception as e:
                print(f"Cache search failed: {e}. Falling back to basic search.")
                self.cache_available = False

        # Final fallback: basic search
        return self._basic_search(query)

    def _ml_search(self, query: str) -> list[dict]:
        """ML-powered search (may fail)"""
        if random.random() < 0.3:  # Simulate failure
            raise ConnectionError("ML service unavailable")
        return [{"result": f"ML result for '{query}'", "score": 0.95}]

    def _cached_search(self, query: str) -> list[dict]:
        """Cached search results"""
        if random.random() < 0.2:  # Simulate failure
            raise ConnectionError("Cache unavailable")
        return [{"result": f"Cached result for '{query}'", "score": 0.85}]

    def _basic_search(self, query: str) -> list[dict]:
        """Basic search that always works"""
        return [{"result": f"Basic result for '{query}'", "score": 0.70}]


# 2. BULKHEAD PATTERN
class ResourcePool:
    """
    Bulkhead pattern: Isolate resources to prevent one failing component
    from consuming all resources and affecting other components.
    """

    def __init__(self, total_capacity: int):
        self.total_capacity = total_capacity
        self.pools = {}

    def create_pool(self, name: str, capacity: int):
        """Create an isolated resource pool"""
        if sum(p["capacity"] for p in self.pools.values()) + capacity > self.total_capacity:
            raise ValueError("Not enough capacity for new pool")

        self.pools[name] = {
            "capacity": capacity,
            "used": 0
        }

    def acquire(self, pool_name: str) -> bool:
        """Acquire a resource from the pool"""
        pool = self.pools.get(pool_name)
        if not pool:
            return False

        if pool["used"] < pool["capacity"]:
            pool["used"] += 1
            return True
        return False

    def release(self, pool_name: str):
        """Release a resource back to the pool"""
        pool = self.pools.get(pool_name)
        if pool and pool["used"] > 0:
            pool["used"] -= 1

    def get_status(self) -> dict:
        """Get status of all pools"""
        return {
            name: f"{pool['used']}/{pool['capacity']}"
            for name, pool in self.pools.items()
        }


# 3. CHAOS ENGINEERING
class ChaosMonkey:
    """
    Chaos engineering: Intentionally inject failures to test system resilience.
    This helps identify weaknesses before they cause production issues.
    """

    def __init__(self, failure_rate: float = 0.1):
        self.failure_rate = failure_rate
        self.enabled = False

    def enable(self):
        """Enable chaos testing"""
        self.enabled = True
        print("🐒 Chaos Monkey enabled!")

    def disable(self):
        """Disable chaos testing"""
        self.enabled = False
        print("🐒 Chaos Monkey disabled")

    def should_inject_failure(self) -> bool:
        """Determine if a failure should be injected"""
        return self.enabled and random.random() < self.failure_rate

    def chaos_wrapper(self, func: Callable) -> Callable:
        """Wrap a function to randomly inject failures"""
        def wrapper(*args, **kwargs):
            if self.should_inject_failure():
                failure_type = random.choice(["timeout", "error", "latency"])

                if failure_type == "timeout":
                    raise TimeoutError("Chaos Monkey: Simulated timeout")
                elif failure_type == "error":
                    raise RuntimeError("Chaos Monkey: Simulated error")
                else:  # latency
                    time.sleep(2)
                    print("🐒 Chaos Monkey: Added 2s latency")

            return func(*args, **kwargs)
        return wrapper


# 4. HEALTH CHECKS
class DistributedService:
    """
    Service with comprehensive health checks to detect issues early.
    """

    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.dependencies_healthy = True

    def health_check(self) -> HealthCheck:
        """Comprehensive health check"""
        # Check uptime
        uptime = time.time() - self.start_time
        if uptime < 10:
            return HealthCheck(
                ServiceHealth.DEGRADED,
                "Service recently started, warming up",
                time.time()
            )

        # Check the error rate
        if self.request_count > 0:
            error_rate = self.error_count / self.request_count
            if error_rate > 0.5:
                return HealthCheck(
                    ServiceHealth.UNHEALTHY,
                    f"High error rate: {error_rate:.2%}",
                    time.time()
                )
            elif error_rate > 0.1:
                return HealthCheck(
                    ServiceHealth.DEGRADED,
                    f"Elevated error rate: {error_rate:.2%}",
                    time.time()
                )

        # Check dependencies
        if not self.dependencies_healthy:
            return HealthCheck(
                ServiceHealth.DEGRADED,
                "Dependencies unavailable",
                time.time()
            )

        return HealthCheck(
            ServiceHealth.HEALTHY,
            "All systems operational",
            time.time()
        )

    def process_request(self) -> str:
        """Process a request and track metrics"""
        self.request_count += 1

        # Simulate occasional failures
        if random.random() < 0.05:
            self.error_count += 1
            raise RuntimeError("Request processing failed")

        return "Success"


# 5. DEFENSIVE PROGRAMMING
class DataProcessor:
    """
    Defensive programming: Validate inputs, handle edge cases,
    and fail gracefully with informative errors.
    """

    @staticmethod
    def process_user_data(data: Any) -> dict:
        """Process user data with defensive validation"""
        # Validate input type
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")

        # Validate required fields
        required_fields = ["user_id", "email"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Validate field types
        if not isinstance(data["user_id"], (int, str)):
            raise TypeError("user_id must be int or str")

        if not isinstance(data["email"], str) or "@" not in data["email"]:
            raise ValueError("Invalid email format")

        # Sanitize optional fields with defaults
        processed = {
            "user_id": str(data["user_id"]),
            "email": data["email"].lower().strip(),
            "name": data.get("name", "Unknown").strip(),
            "age": max(0, min(150, data.get("age", 0))),  # Clamp to valid range
            "preferences": data.get("preferences", {}),
        }

        return processed

    @staticmethod
    def safe_divide(a: float, b: float, default: float = 0.0) -> float:
        """Safely divide with fallback for edge cases"""
        try:
            if b == 0:
                return default
            result = a / b
            # Check for infinity or NaN
            if not (-float('inf') < result < float('inf')):
                return default
            return result
        except (TypeError, ValueError, OverflowError):
            return default


def demonstrate_graceful_degradation():
    """Demonstrate graceful degradation"""
    print("\n=== GRACEFUL DEGRADATION ===")
    service = SearchService()

    # Perform multiple searches, showing degradation
    for i in range(5):
        results = service.search(f"query_{i}")
        print(f"Search {i+1}: {results[0]}")


def demonstrate_bulkhead():
    """Demonstrate a bulkhead pattern"""
    print("\n=== BULKHEAD PATTERN ===")
    pool = ResourcePool(total_capacity=10)

    # Create isolated pools for different services
    pool.create_pool("api", capacity=5)
    pool.create_pool("background_jobs", capacity=3)
    pool.create_pool("admin", capacity=2)

    print(f"Initial state: {pool.get_status()}")

    # Simulate API service consuming all its resources
    print("\nAPI service under heavy load...")
    for _ in range(5):
        pool.acquire("api")
    print(f"After API load: {pool.get_status()}")

    # Background jobs can still work
    if pool.acquire("background_jobs"):
        print("✓ Background jobs still have resources (bulkhead working!)")


def demonstrate_chaos_engineering():
    """Demonstrate chaos engineering"""
    print("\n=== CHAOS ENGINEERING ===")
    chaos = ChaosMonkey(failure_rate=0.5)

    def critical_operation():
        return "Operation completed"

    # Wrap function with chaos monkey
    chaotic_operation = chaos.chaos_wrapper(critical_operation)

    chaos.enable()

    # Try the operation multiple times
    for i in range(5):
        try:
            result = chaotic_operation()
            print(f"Attempt {i+1}: {result}")
        except Exception as e:
            print(f"Attempt {i+1}: Failed - {e}")


def demonstrate_health_checks():
    """Demonstrate health checks"""
    print("\n=== HEALTH CHECKS ===")
    service = DistributedService()

    # Check health at different stages
    health = service.health_check()
    print(f"Initial: {health.status.value} - {health.message}")

    # Simulate some requests
    time.sleep(0.5)
    for _ in range(20):
        try:
            service.process_request()
        except RuntimeError:
            pass

    health = service.health_check()
    print(f"After load: {health.status.value} - {health.message}")


def demonstrate_defensive_programming():
    """Demonstrate defensive programming"""
    print("\n=== DEFENSIVE PROGRAMMING ===")
    processor = DataProcessor()

    # Valid data
    try:
        result = processor.process_user_data({
            "user_id": 123,
            "email": "  USER@EXAMPLE.COM  ",
            "age": 25
        })
        print(f"✓ Valid data processed: {result}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Invalid data
    try:
        processor.process_user_data({
            "user_id": 123,
            # Missing email
        })
    except ValueError as e:
        print(f"✓ Caught missing field: {e}")

    # Edge cases in division
    print(f"\n10 / 2 = {processor.safe_divide(10, 2)}")
    print(f"10 / 0 = {processor.safe_divide(10, 0)} (default)")
    print(f"inf / inf = {processor.safe_divide(float('inf'), float('inf'), default=-1)} (default)")


def main():
    """Run all demonstrations"""
    print("ROBUSTNESS PATTERNS IN DISTRIBUTED SYSTEMS")
    print("=" * 50)

    demonstrate_graceful_degradation()
    demonstrate_bulkhead()
    demonstrate_chaos_engineering()
    demonstrate_health_checks()
    demonstrate_defensive_programming()

    print("\n" + "=" * 50)
    print("Key Takeaways:")
    print("1. Design for failure - assume components will fail")
    print("2. Isolate failures - prevent cascade effects")
    print("3. Monitor health - detect issues early")
    print("4. Test resilience - use chaos engineering")
    print("5. Validate everything - defensive programming saves lives")


if __name__ == "__main__":
    main()
