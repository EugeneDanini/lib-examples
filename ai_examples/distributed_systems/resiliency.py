"""
Resiliency Patterns in Distributed Systems

Resiliency is the ability to recover quickly from failures and continue operating.
It combines fault tolerance, graceful degradation, and self-healing capabilities.

Key resiliency patterns:
1. Retry with Exponential Backoff - Automatically retry failed operations
2. Timeout Pattern - Prevent indefinite waiting
3. Bulkhead Isolation - Limit failure blast radius
4. Fallback Strategy - Provide alternative responses
5. Self-Healing - Automatic detection and recovery
"""

import time
import random
from typing import Any, Callable, Optional, TypeVar
from dataclasses import dataclass, field
from enum import Enum


T = TypeVar('T')


# 1. RETRY WITH EXPONENTIAL BACKOFF
@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class RetryResult:
    """Result of a retry operation"""
    success: bool
    attempts: int
    total_time: float
    result: Any = None
    error: Optional[Exception] = None


class RetryStrategy:
    """
    Retry with exponential backoff automatically retries failed operations.
    Delays increase exponentially to avoid overwhelming failing services.
    """

    def __init__(self, config: RetryConfig = RetryConfig()):
        self.config = config
        self.total_attempts = 0
        self.total_retries = 0
        self.success_count = 0
        self.failure_count = 0

    def execute(self, func: Callable[[], T], *args, **kwargs) -> RetryResult:
        """Execute a function with retry logic"""
        attempts = 0
        start_time = time.time()
        last_error = None

        while attempts < self.config.max_attempts:
            attempts += 1
            self.total_attempts += 1

            try:
                result = func(*args, **kwargs)
                self.success_count += 1
                total_time = time.time() - start_time

                if attempts > 1:
                    print(f"    [RETRY] Succeeded on attempt {attempts}")

                return RetryResult(
                    success=True,
                    attempts=attempts,
                    total_time=total_time,
                    result=result
                )

            except Exception as e:
                last_error = e
                self.total_retries += 1

                if attempts < self.config.max_attempts:
                    delay = self._calculate_delay(attempts)
                    print(f"    [RETRY] Attempt {attempts} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    print(f"    [RETRY] All {attempts} attempts failed")

        self.failure_count += 1
        total_time = time.time() - start_time

        return RetryResult(
            success=False,
            attempts=attempts,
            total_time=total_time,
            error=last_error
        )

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry using exponential backoff"""
        delay = min(
            self.config.initial_delay * (self.config.exponential_base ** (attempt - 1)),
            self.config.max_delay
        )

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            delay = delay * (0.5 + random.random() * 0.5)

        return delay

    def get_stats(self) -> dict:
        """Get retry statistics"""
        success_rate = (self.success_count / self.total_attempts * 100) if self.total_attempts > 0 else 0
        return {
            "total_attempts": self.total_attempts,
            "total_retries": self.total_retries,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": f"{success_rate:.1f}%"
        }


# 2. TIMEOUT PATTERN
class TimeoutException(Exception):
    """Exception raised when operation times out"""
    pass


@dataclass
class TimeoutConfig:
    """Timeout configuration"""
    default_timeout: float = 5.0
    long_timeout: float = 30.0
    short_timeout: float = 1.0


class TimeoutManager:
    """
    Timeout pattern prevents indefinite waiting.
    Ensures operations fail fast rather than hanging forever.
    """

    def __init__(self, config: TimeoutConfig = TimeoutConfig()):
        self.config = config
        self.timeout_count = 0
        self.success_count = 0

    def execute_with_timeout(
        self,
        func: Callable[[], T],
        timeout: Optional[float] = None,
        *args,
        **kwargs
    ) -> T:
        """Execute a function with timeout"""
        timeout = timeout or self.config.default_timeout
        start_time = time.time()

        try:
            # Note: Real implementation would use threading or asyncio
            # This is simplified for demonstration
            result = func(*args, **kwargs)

            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.timeout_count += 1
                raise TimeoutException(f"Operation exceeded timeout of {timeout}s (took {elapsed:.2f}s)")

            self.success_count += 1
            return result

        except TimeoutException:
            raise
        except Exception as e:
            # Wrap other exceptions
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.timeout_count += 1
                raise TimeoutException(f"Operation timed out after {elapsed:.2f}s")
            raise e

    def with_short_timeout(self, func: Callable[[], T], *args, **kwargs) -> T:
        """Execute with a short timeout (for fast operations)"""
        return self.execute_with_timeout(func, self.config.short_timeout, *args, **kwargs)

    def with_long_timeout(self, func: Callable[[], T], *args, **kwargs) -> T:
        """Execute with a long timeout (for slow operations)"""
        return self.execute_with_timeout(func, self.config.long_timeout, *args, **kwargs)

    def get_stats(self) -> dict:
        """Get timeout statistics"""
        total = self.success_count + self.timeout_count
        timeout_rate = (self.timeout_count / total * 100) if total > 0 else 0
        return {
            "success_count": self.success_count,
            "timeout_count": self.timeout_count,
            "timeout_rate": f"{timeout_rate:.1f}%"
        }


# 3. BULKHEAD ISOLATION
class ResourceType(Enum):
    """Types of resources"""
    CPU = "cpu"
    MEMORY = "memory"
    CONNECTIONS = "connections"
    THREADS = "threads"


@dataclass
class BulkheadCompartment:
    """An isolated resource compartment"""
    name: str
    resource_type: ResourceType
    capacity: int
    in_use: int = 0
    waiting: int = 0
    total_acquisitions: int = 0
    total_rejections: int = 0

    def try_acquire(self) -> bool:
        """Try to acquire a resource from this compartment"""
        if self.in_use < self.capacity:
            self.in_use += 1
            self.total_acquisitions += 1
            return True
        else:
            self.waiting += 1
            self.total_rejections += 1
            return False

    def release(self):
        """Release a resource back to the compartment"""
        if self.in_use > 0:
            self.in_use -= 1
        if self.waiting > 0:
            self.waiting -= 1

    def get_utilization(self) -> float:
        """Get current utilization percentage"""
        return (self.in_use / self.capacity * 100) if self.capacity > 0 else 0


class BulkheadIsolation:
    """
    Bulkhead isolation limits failure blast radius.
    Isolates resources so one failing component can't affect others.
    """

    def __init__(self):
        self.compartments: dict[str, BulkheadCompartment] = {}

    def create_compartment(self, name: str, resource_type: ResourceType, capacity: int):
        """Create an isolated compartment"""
        compartment = BulkheadCompartment(name, resource_type, capacity)
        self.compartments[name] = compartment
        print(f"  [BULKHEAD] Created compartment '{name}' ({resource_type.value}, capacity={capacity})")

    def acquire(self, compartment_name: str, timeout: float = 1.0) -> bool:
        """Acquire a resource from a compartment"""
        if compartment_name not in self.compartments:
            return False

        compartment = self.compartments[compartment_name]
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            if compartment.try_acquire():
                return True
            time.sleep(0.01)

        return False

    def release(self, compartment_name: str):
        """Release a resource back to a compartment"""
        if compartment_name in self.compartments:
            self.compartments[compartment_name].release()

    def get_status(self) -> dict:
        """Get status of all compartments"""
        return {
            name: {
                "in_use": comp.in_use,
                "capacity": comp.capacity,
                "utilization": f"{comp.get_utilization():.1f}%",
                "waiting": comp.waiting,
                "rejections": comp.total_rejections
            }
            for name, comp in self.compartments.items()
        }


# 4. FALLBACK STRATEGY
@dataclass
class FallbackChain:
    """A chain of fallback strategies"""
    primary: Callable
    fallbacks: list[Callable] = field(default_factory=list)
    name: str = "fallback_chain"


class FallbackStrategy:
    """
    Fallback strategy provides alternative responses when primary fails.
    Ensures the system continues operating with degraded functionality.
    """

    def __init__(self):
        self.primary_success = 0
        self.fallback_used = 0
        self.total_failures = 0

    def execute_with_fallback(
        self,
        primary: Callable[[], T],
        fallback: Callable[[], T],
        fallback_name: str = "fallback"
    ) -> T:
        """Execute primary, fallback to alternative on failure"""
        try:
            result = primary()
            self.primary_success += 1
            return result
        except Exception as e:
            print(f"    [FALLBACK] Primary failed: {e}. Using {fallback_name}...")
            self.fallback_used += 1
            try:
                return fallback()
            except Exception as fallback_error:
                self.total_failures += 1
                print(f"    [FALLBACK] {fallback_name} also failed: {fallback_error}")
                raise

    def execute_chain(self, chain: FallbackChain) -> Any:
        """Execute a chain of fallbacks"""
        # Try primary
        try:
            result = chain.primary()
            self.primary_success += 1
            return result
        except Exception as e:
            print(f"    [FALLBACK] Primary failed: {e}")

        # Try each fallback in order
        for i, fallback in enumerate(chain.fallbacks):
            try:
                print(f"    [FALLBACK] Trying fallback {i+1}/{len(chain.fallbacks)}...")
                result = fallback()
                self.fallback_used += 1
                return result
            except Exception as e:
                print(f"    [FALLBACK] Fallback {i+1} failed: {e}")
                continue

        # All fallbacks failed
        self.total_failures += 1
        raise RuntimeError("All fallback strategies failed")

    def get_stats(self) -> dict:
        """Get fallback statistics"""
        total_attempts = self.primary_success + self.fallback_used + self.total_failures
        primary_success_rate = (self.primary_success / total_attempts * 100) if total_attempts > 0 else 0
        fallback_rate = (self.fallback_used / total_attempts * 100) if total_attempts > 0 else 0

        return {
            "primary_success": self.primary_success,
            "fallback_used": self.fallback_used,
            "total_failures": self.total_failures,
            "primary_success_rate": f"{primary_success_rate:.1f}%",
            "fallback_rate": f"{fallback_rate:.1f}%"
        }


# 5. SELF-HEALING
class ComponentHealth(Enum):
    """Health status of a component"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class Component:
    """A system component that can be monitored and healed"""
    name: str
    health: ComponentHealth = ComponentHealth.HEALTHY
    failure_count: int = 0
    last_health_check: float = field(default_factory=time.time)
    recovery_attempts: int = 0


class SelfHealingSystem:
    """
    Self-healing system automatically detects and recovers from failures.
    Reduces manual intervention and improves system resilience.
    """

    def __init__(
        self,
        health_check_interval: float = 5.0,
        failure_threshold: int = 3,
        recovery_cooldown: float = 10.0
    ):
        self.health_check_interval = health_check_interval
        self.failure_threshold = failure_threshold
        self.recovery_cooldown = recovery_cooldown

        self.components: dict[str, Component] = {}
        self.recovery_handlers: dict[str, Callable] = {}
        self.total_recoveries = 0

    def register_component(self, name: str, recovery_handler: Callable):
        """Register a component for monitoring"""
        component = Component(name=name)
        self.components[name] = component
        self.recovery_handlers[name] = recovery_handler
        print(f"  [SELF-HEAL] Registered component '{name}'")

    def report_failure(self, component_name: str):
        """Report a component failure"""
        if component_name not in self.components:
            return

        component = self.components[component_name]
        component.failure_count += 1

        print(f"  [SELF-HEAL] Failure reported for '{component_name}' (count: {component.failure_count})")

        # Check if a threshold exceeded
        if component.failure_count >= self.failure_threshold:
            component.health = ComponentHealth.FAILED
            print(f"  [SELF-HEAL] Component '{component_name}' marked as FAILED")
            self._trigger_recovery(component_name)

    def _trigger_recovery(self, component_name: str):
        """Trigger automatic recovery for a failed component"""
        component = self.components[component_name]

        if component.health != ComponentHealth.FAILED:
            return

        # Check recovery cooldown
        time_since_check = time.time() - component.last_health_check
        if time_since_check < self.recovery_cooldown:
            print(f"  [SELF-HEAL] Recovery cooldown active for '{component_name}'")
            return

        print(f"  [SELF-HEAL] Attempting recovery for '{component_name}'...")
        component.health = ComponentHealth.RECOVERING
        component.recovery_attempts += 1

        try:
            # Execute recovery handler
            recovery_handler = self.recovery_handlers.get(component_name)
            if recovery_handler:
                recovery_handler()

            # Mark as recovered
            component.health = ComponentHealth.HEALTHY
            component.failure_count = 0
            component.last_health_check = time.time()
            self.total_recoveries += 1
            print(f"  [SELF-HEAL] Component '{component_name}' recovered successfully")

        except Exception as e:
            component.health = ComponentHealth.FAILED
            print(f"  [SELF-HEAL] Recovery failed for '{component_name}': {e}")

    def health_check(self, component_name: str) -> ComponentHealth:
        """Perform health check on a component"""
        if component_name not in self.components:
            return ComponentHealth.FAILED

        component = self.components[component_name]
        component.last_health_check = time.time()

        return component.health

    def get_system_health(self) -> dict:
        """Get overall system health"""
        healthy = sum(1 for c in self.components.values() if c.health == ComponentHealth.HEALTHY)
        failed = sum(1 for c in self.components.values() if c.health == ComponentHealth.FAILED)
        total = len(self.components)

        return {
            "total_components": total,
            "healthy": healthy,
            "failed": failed,
            "total_recoveries": self.total_recoveries,
            "health_percentage": f"{(healthy/total*100):.1f}%" if total > 0 else "0%"
        }

    def get_component_status(self) -> dict:
        """Get status of all components"""
        return {
            name: {
                "health": comp.health.value,
                "failure_count": comp.failure_count,
                "recovery_attempts": comp.recovery_attempts
            }
            for name, comp in self.components.items()
        }


def demonstrate_retry_backoff():
    """Demonstrate retry with exponential backoff"""
    print("\n=== RETRY WITH EXPONENTIAL BACKOFF ===")

    retry = RetryStrategy(RetryConfig(max_attempts=4, initial_delay=0.1, max_delay=1.0))

    # Simulate unreliable service
    attempt_counter = 0
    def unreliable_service():
        nonlocal attempt_counter
        attempt_counter += 1
        if attempt_counter < 3:
            raise ConnectionError("Service unavailable")
        return "Success!"

    print("\nCalling unreliable service:")
    result = retry.execute(unreliable_service)
    print(f"Result: success={result.success}, attempts={result.attempts}, time={result.total_time:.2f}s")

    # Test with a service that always fails
    print("\nCalling service that always fails:")
    result = retry.execute(lambda: 1/0)  # Will raise ZeroDivisionError
    print(f"Result: success={result.success}, attempts={result.attempts}")

    print(f"\nRetry stats: {retry.get_stats()}")


def demonstrate_timeout_pattern():
    """Demonstrate a timeout pattern"""
    print("\n=== TIMEOUT PATTERN ===")

    timeout_mgr = TimeoutManager(TimeoutConfig(default_timeout=0.5, short_timeout=0.1))

    # Fast operation
    print("\nExecuting fast operation:")
    try:
        result = timeout_mgr.execute_with_timeout(lambda: "Quick result", timeout=1.0)
        print(f"  Result: {result}")
    except TimeoutException as e:
        print(f"  Timeout: {e}")

    # Slow operation that times out
    print("\nExecuting slow operation with short timeout:")
    try:
        def slow_operation():
            time.sleep(0.2)
            return "Slow result"

        result = timeout_mgr.with_short_timeout(slow_operation)
        print(f"  Result: {result}")
    except TimeoutException as e:
        print(f"  Timeout: {e}")

    print(f"\nTimeout stats: {timeout_mgr.get_stats()}")


def demonstrate_bulkhead_isolation():
    """Demonstrate bulkhead isolation"""
    print("\n=== BULKHEAD ISOLATION ===")

    bulkhead = BulkheadIsolation()

    # Create compartments for different services
    print("\nCreating isolated compartments:")
    bulkhead.create_compartment("user_service", ResourceType.CONNECTIONS, capacity=5)
    bulkhead.create_compartment("payment_service", ResourceType.CONNECTIONS, capacity=3)
    bulkhead.create_compartment("reporting", ResourceType.THREADS, capacity=2)

    # Simulate user service consuming all resources
    print("\nUser service under heavy load (acquiring all connections):")
    for i in range(5):
        bulkhead.acquire("user_service")

    print(f"Status after user service load: {bulkhead.get_status()}")

    # Payment service still has resources
    print("\nPayment service can still operate:")
    if bulkhead.acquire("payment_service"):
        print("  ✓ Payment service acquired connection")

    # Try to exceed capacity
    print("\nTrying to exceed user service capacity:")
    if not bulkhead.acquire("user_service", timeout=0.1):
        print("  ✗ Request rejected - compartment full (isolation working!)")

    print(f"\nFinal status: {bulkhead.get_status()}")


def demonstrate_fallback_strategy():
    """Demonstrate a fallback strategy"""
    print("\n=== FALLBACK STRATEGY ===")

    fallback = FallbackStrategy()

    # Primary with a simple fallback
    print("\nPrimary service with fallback:")

    def primary_service():
        if random.random() < 0.5:
            raise RuntimeError("Primary service unavailable")
        return {"source": "primary", "data": "fresh data"}

    def fallback_cache():
        return {"source": "cache", "data": "cached data"}

    for i in range(3):
        result = fallback.execute_with_fallback(primary_service, fallback_cache, "cache")
        print(f"  Attempt {i+1}: {result}")

    # Fallback chain
    print("\nFallback chain (primary -> cache -> static):")

    def primary_db():
        raise RuntimeError("Database down")

    def cache_service():
        raise RuntimeError("Cache down")

    def static_fallback():
        return {"source": "static", "data": "default value"}

    chain = FallbackChain(
        primary=primary_db,
        fallbacks=[cache_service, static_fallback],
        name="data_chain"
    )

    result = fallback.execute_chain(chain)
    print(f"  Final result: {result}")

    print(f"\nFallback stats: {fallback.get_stats()}")


def demonstrate_self_healing():
    """Demonstrate self-healing"""
    print("\n=== SELF-HEALING ===")

    healing_system = SelfHealingSystem(
        failure_threshold=3,
        recovery_cooldown=1.0
    )

    # Register components with recovery handlers
    print("\nRegistering components:")

    def recover_database():
        print("    → Restarting database connection pool")
        time.sleep(0.1)

    def recover_cache():
        print("    → Clearing and reinitializing cache")
        time.sleep(0.1)

    healing_system.register_component("database", recover_database)
    healing_system.register_component("cache", recover_cache)

    # Simulate failures
    print("\nSimulating failures:")
    healing_system.report_failure("database")
    healing_system.report_failure("database")
    healing_system.report_failure("database")  # Triggers recovery

    time.sleep(0.5)

    # Check health
    print("\nSystem health check:")
    health = healing_system.get_system_health()
    print(f"  {health}")

    print("\nComponent status:")
    for name, status in healing_system.get_component_status().items():
        print(f"  {name}: {status}")


def main():
    """Run all demonstrations"""
    print("RESILIENCY PATTERNS IN DISTRIBUTED SYSTEMS")
    print("=" * 60)

    demonstrate_retry_backoff()
    demonstrate_timeout_pattern()
    demonstrate_bulkhead_isolation()
    demonstrate_fallback_strategy()
    demonstrate_self_healing()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("1. Retry with backoff - Automatically retry with increasing delays")
    print("2. Timeout pattern - Fail fast rather than hang indefinitely")
    print("3. Bulkhead isolation - Limit failure blast radius")
    print("4. Fallback strategy - Provide alternatives when primary fails")
    print("5. Self-healing - Automatic detection and recovery from failures")
    print("\nResiliency = Fail gracefully + Recover automatically + Learn from failures")


if __name__ == "__main__":
    main()
