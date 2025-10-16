"""
Availability Patterns in Distributed Systems

Availability is the percentage of time a system is operational and accessible.
High-availability (HA) systems aim for minimal downtime (e.g., 99.99% = ~52 minutes/year).

Key availability patterns:
1. Redundancy - Multiple instances to eliminate single points of failure
2. Failover - Automatic switching to back-up when primary fails
3. Replication - Data copies across multiple nodes
4. Health Monitoring - Detect and respond to failures quickly
5. Circuit Breaker - Prevent cascading failures
"""

import time
import random
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum


# 1. REDUNDANCY
class NodeStatus(Enum):
    """Status of a node"""
    ACTIVE = "active"
    STANDBY = "standby"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class ServiceNode:
    """A service node with redundancy support"""
    node_id: str
    status: NodeStatus = NodeStatus.ACTIVE
    last_heartbeat: float = field(default_factory=time.time)
    request_count: int = 0

    def is_healthy(self) -> bool:
        """Check if the node is healthy"""
        # Consider unhealthy if no heartbeat in the last 5 seconds
        return (time.time() - self.last_heartbeat) < 5 and self.status != NodeStatus.FAILED

    def heartbeat(self):
        """Update heartbeat timestamp"""
        self.last_heartbeat = time.time()

    def process_request(self, request: str) -> str:
        """Process a request"""
        if not self.is_healthy():
            raise RuntimeError(f"Node {self.node_id} is not healthy")

        self.request_count += 1
        return f"Node {self.node_id} processed: {request}"


class RedundantService:
    """
    Service with redundancy - multiple nodes to eliminate the single point of failure.
    If one node fails, others continue serving requests.
    """

    def __init__(self, num_nodes: int = 3):
        self.nodes = [
            ServiceNode(node_id=f"node_{i}")
            for i in range(num_nodes)
        ]
        self.primary_index = 0

    def get_healthy_nodes(self) -> list[ServiceNode]:
        """Get the list of healthy nodes"""
        return [node for node in self.nodes if node.is_healthy()]

    def select_node(self) -> Optional[ServiceNode]:
        """Select a healthy node to handle the request"""
        healthy_nodes = self.get_healthy_nodes()
        if not healthy_nodes:
            return None

        # Round-robin across healthy nodes
        node = healthy_nodes[self.primary_index % len(healthy_nodes)]
        self.primary_index += 1
        return node

    def process_request(self, request: str) -> str:
        """Process request with automatic failover to healthy nodes"""
        node = self.select_node()
        if not node:
            return "Error: No healthy nodes available"

        try:
            return node.process_request(request)
        except Exception as e:
            # Mark the node as failed and retry with another node
            node.status = NodeStatus.FAILED
            retry_node = self.select_node()
            if retry_node:
                return retry_node.process_request(request)
            return f"Error: Request failed - {e}"

    def get_availability(self) -> float:
        """Calculate current availability percentage"""
        if not self.nodes:
            return 0.0
        healthy_count = len(self.get_healthy_nodes())
        return (healthy_count / len(self.nodes)) * 100


# 2. FAILOVER
class FailoverState(Enum):
    """State of a failover system"""
    PRIMARY_ACTIVE = "primary_active"
    SECONDARY_ACTIVE = "secondary_active"
    FAILING_OVER = "failing_over"
    BOTH_FAILED = "both_failed"


@dataclass
class DatabaseNode:
    """Database node (primary or secondary)"""
    node_id: str
    is_primary: bool
    is_available: bool = True
    data: dict = field(default_factory=dict)

    def write(self, key: str, value: Any):
        """Write data"""
        if not self.is_available:
            raise RuntimeError(f"{self.node_id} is not available")
        self.data[key] = value

    def read(self, key: str) -> Optional[Any]:
        """Read data"""
        if not self.is_available:
            raise RuntimeError(f"{self.node_id} is not available")
        return self.data.get(key)

    def sync_from(self, other: 'DatabaseNode'):
        """Sync data from another node"""
        self.data = other.data.copy()


class ActivePassiveFailover:
    """
    Active-passive failover: One primary handles requests, secondary takes over on failure.
    Ensures high availability by automatic failover to a standby system.
    """

    def __init__(self):
        self.primary = DatabaseNode("primary", is_primary=True)
        self.secondary = DatabaseNode("secondary", is_primary=False)
        self.state = FailoverState.PRIMARY_ACTIVE
        self.failover_count = 0

    def _check_health(self):
        """Check health and trigger failover if needed"""
        if self.state == FailoverState.PRIMARY_ACTIVE and not self.primary.is_available:
            self._failover_to_secondary()
        elif self.state == FailoverState.SECONDARY_ACTIVE and not self.secondary.is_available:
            self.state = FailoverState.BOTH_FAILED

    def _failover_to_secondary(self):
        """Perform failover to secondary"""
        print(f"  [FAILOVER] Primary failed, switching to secondary...")
        self.state = FailoverState.FAILING_OVER

        # Promote secondary to primary
        self.secondary.is_primary = True
        self.state = FailoverState.SECONDARY_ACTIVE
        self.failover_count += 1
        print(f"  [FAILOVER] Secondary is now active")

    def _get_active_node(self) -> Optional[DatabaseNode]:
        """Get the currently active node"""
        self._check_health()

        if self.state == FailoverState.PRIMARY_ACTIVE:
            return self.primary
        elif self.state == FailoverState.SECONDARY_ACTIVE:
            return self.secondary
        else:
            return None

    def write(self, key: str, value: Any):
        """Write with automatic failover"""
        active_node = self._get_active_node()
        if not active_node:
            raise RuntimeError("No available nodes for write")

        active_node.write(key, value)

        # Replicate to standby (if available)
        standby = self.secondary if self.state == FailoverState.PRIMARY_ACTIVE else self.primary
        if standby.is_available:
            try:
                standby.write(key, value)
            except RuntimeError:
                pass  # Standby replication failure is non-critical

    def read(self, key: str) -> Optional[Any]:
        """Read with automatic failover"""
        active_node = self._get_active_node()
        if not active_node:
            raise RuntimeError("No available nodes for read")

        return active_node.read(key)

    def get_status(self) -> dict:
        """Get failover status"""
        return {
            "state": self.state.value,
            "primary_available": self.primary.is_available,
            "secondary_available": self.secondary.is_available,
            "failover_count": self.failover_count
        }


# 3. REPLICATION
class ReplicationStrategy(Enum):
    """Data replication strategies"""
    SYNCHRONOUS = "synchronous"  # Wait for all replicas
    ASYNCHRONOUS = "asynchronous"  # Don't wait for replicas
    QUORUM = "quorum"  # Wait for a majority


@dataclass
class ReplicaNode:
    """Replica node"""
    replica_id: str
    data: dict = field(default_factory=dict)
    is_online: bool = True
    lag_ms: int = 0  # Replication lag in milliseconds

    def write(self, key: str, value: Any) -> bool:
        """Write to replica"""
        if not self.is_online:
            return False

        # Simulate replication lag
        if self.lag_ms > 0:
            time.sleep(self.lag_ms / 1000)

        self.data[key] = value
        return True

    def read(self, key: str) -> Optional[Any]:
        """Read from replica"""
        if not self.is_online:
            return None
        return self.data.get(key)


class ReplicatedDataStore:
    """
    Data store with replication across multiple nodes.
    Provides high availability and durability through data redundancy.
    """

    def __init__(self, num_replicas: int = 3, strategy: ReplicationStrategy = ReplicationStrategy.QUORUM):
        self.replicas = [
            ReplicaNode(replica_id=f"replica_{i}")
            for i in range(num_replicas)
        ]
        self.strategy = strategy

    def write(self, key: str, value: Any) -> bool:
        """Write data with replication"""
        success_count = 0

        for replica in self.replicas:
            if replica.is_online:
                try:
                    if replica.write(key, value):
                        success_count += 1
                except RuntimeError:
                    pass

        # Check if write meets strategy requirements
        if self.strategy == ReplicationStrategy.SYNCHRONOUS:
            # All replicas must succeed
            return success_count == len(self.replicas)
        elif self.strategy == ReplicationStrategy.QUORUM:
            # Majority must succeed
            return success_count > len(self.replicas) // 2
        else:  # ASYNCHRONOUS
            # At least one must succeed
            return success_count > 0

    def read(self, key: str) -> Optional[Any]:
        """Read from any available replica"""
        for replica in self.replicas:
            if replica.is_online:
                value = replica.read(key)
                if value is not None:
                    return value
        return None

    def get_replication_status(self) -> dict:
        """Get replication status"""
        online_replicas = [r for r in self.replicas if r.is_online]
        return {
            "total_replicas": len(self.replicas),
            "online_replicas": len(online_replicas),
            "replication_factor": f"{len(online_replicas)}/{len(self.replicas)}",
            "strategy": self.strategy.value
        }


# 4. HEALTH MONITORING
@dataclass
class HealthMetrics:
    """Health metrics for monitoring"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    response_time_ms: float
    error_rate: float

    def is_healthy(self) -> bool:
        """Determine if metrics indicate a healthy state"""
        return (
            self.cpu_usage < 80 and
            self.memory_usage < 85 and
            self.response_time_ms < 1000 and
            self.error_rate < 0.05
        )


class HealthMonitor:
    """
    Continuous health monitoring to detect failures quickly.
    Early detection enables faster recovery and higher availability.
    """

    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self.services: dict[str, Callable] = {}
        self.health_history: dict[str, list[HealthMetrics]] = {}
        self.alerts: list[str] = []

    def register_service(self, service_name: str, health_check_fn: Callable):
        """Register a service for monitoring"""
        self.services[service_name] = health_check_fn
        self.health_history[service_name] = []

    def check_service(self, service_name: str) -> Optional[HealthMetrics]:
        """Check health of a specific service"""
        if service_name not in self.services:
            return None

        try:
            metrics = self.services[service_name]()
            self.health_history[service_name].append(metrics)

            # Keep only the last 10 measurements
            if len(self.health_history[service_name]) > 10:
                self.health_history[service_name].pop(0)

            # Generate alerts if unhealthy
            if not metrics.is_healthy():
                alert = f"[ALERT] {service_name} is unhealthy: CPU={metrics.cpu_usage:.1f}%, Mem={metrics.memory_usage:.1f}%, RT={metrics.response_time_ms:.0f}ms, Err={metrics.error_rate:.2%}"
                self.alerts.append(alert)

            return metrics
        except Exception as e:
            alert = f"[ALERT] {service_name} health check failed: {e}"
            self.alerts.append(alert)
            return None

    def check_all_services(self) -> dict[str, HealthMetrics]:
        """Check the health of all registered services"""
        results = {}
        for service_name in self.services:
            metrics = self.check_service(service_name)
            if metrics:
                results[service_name] = metrics
        return results

    def get_alerts(self) -> list[str]:
        """Get recent alerts"""
        alerts = self.alerts.copy()
        self.alerts.clear()
        return alerts


# 5. CIRCUIT BREAKER
class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker pattern prevents cascading failures.
    Stops making requests to failing service, giving it time to recover.
    """

    def __init__(self, failure_threshold: int = 5, timeout: float = 5.0, success_threshold: int = 2):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self.last_failure_time and (time.time() - self.last_failure_time) >= self.timeout:
                print("  [CIRCUIT BREAKER] Entering half-open state (testing recovery)")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise RuntimeError("Circuit breaker is OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle a successful call"""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                print("  [CIRCUIT BREAKER] Service recovered, closing circuit")
                self.state = CircuitState.CLOSED
                self.success_count = 0

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            print("  [CIRCUIT BREAKER] Recovery failed, opening circuit")
            self.state = CircuitState.OPEN
            self.failure_count = 0
        elif self.failure_count >= self.failure_threshold:
            print(f"  [CIRCUIT BREAKER] Failure threshold reached ({self.failure_count}), opening circuit")
            self.state = CircuitState.OPEN

    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state


def demonstrate_redundancy():
    """Demonstrate a redundancy pattern"""
    print("\n=== REDUNDANCY ===")

    service = RedundantService(num_nodes=3)

    # Process requests normally
    print("\nProcessing requests with all nodes healthy:")
    for i in range(3):
        result = service.process_request(f"request_{i}")
        print(f"  {result}")

    print(f"Availability: {service.get_availability():.1f}%")

    # Simulate node failure
    print("\nSimulating node_1 failure:")
    service.nodes[1].status = NodeStatus.FAILED

    # Requests still work with remaining nodes
    for i in range(3):
        result = service.process_request(f"request_{i+3}")
        print(f"  {result}")

    print(f"Availability: {service.get_availability():.1f}%")


def demonstrate_failover():
    """Demonstrate active-passive failover"""
    print("\n=== FAILOVER ===")

    db = ActivePassiveFailover()

    # Normal operations
    print("\nNormal operations (primary active):")
    db.write("user:1", {"name": "Alice"})
    db.write("user:2", {"name": "Bob"})
    print(f"  Wrote 2 users")
    print(f"  Read user:1: {db.read('user:1')}")
    print(f"  Status: {db.get_status()}")

    # Simulate primary failure
    print("\nSimulating primary database failure:")
    db.primary.is_available = False

    # Automatic failover
    db.write("user:3", {"name": "Charlie"})
    print(f"  Wrote user:3 (via secondary)")
    print(f"  Read user:1: {db.read('user:1')}")
    print(f"  Status: {db.get_status()}")


def demonstrate_replication():
    """Demonstrate data replication"""
    print("\n=== REPLICATION ===")

    # Quorum-based replication
    store = ReplicatedDataStore(num_replicas=5, strategy=ReplicationStrategy.QUORUM)

    print("\nWriting with quorum replication (need 3/5 replicas):")
    success = store.write("config:timeout", 30)
    print(f"  Write successful: {success}")
    print(f"  Status: {store.get_replication_status()}")

    # Simulate replica failures
    print("\nSimulating 2 replica failures:")
    store.replicas[0].is_online = False
    store.replicas[1].is_online = False

    # Can still write (3/5 replicas available)
    success = store.write("config:retry", 3)
    print(f"  Write successful: {success}")
    print(f"  Status: {store.get_replication_status()}")

    # Can still read
    value = store.read("config:timeout")
    print(f"  Read config:timeout: {value}")


def demonstrate_health_monitoring():
    """Demonstrate health monitoring"""
    print("\n=== HEALTH MONITORING ===")

    monitor = HealthMonitor()

    # Register mock services
    def api_health_check():
        return HealthMetrics(
            timestamp=time.time(),
            cpu_usage=random.uniform(20, 90),
            memory_usage=random.uniform(30, 90),
            response_time_ms=random.uniform(50, 500),
            error_rate=random.uniform(0, 0.1)
        )

    def db_health_check():
        return HealthMetrics(
            timestamp=time.time(),
            cpu_usage=random.uniform(30, 85),
            memory_usage=random.uniform(40, 95),
            response_time_ms=random.uniform(10, 200),
            error_rate=random.uniform(0, 0.05)
        )

    monitor.register_service("api", api_health_check)
    monitor.register_service("database", db_health_check)

    # Check health
    print("\nPerforming health checks:")
    results = monitor.check_all_services()
    for service_name, metrics in results.items():
        status = "✓ Healthy" if metrics.is_healthy() else "✗ Unhealthy"
        print(f"  {service_name}: {status} (CPU={metrics.cpu_usage:.1f}%, Mem={metrics.memory_usage:.1f}%, RT={metrics.response_time_ms:.0f}ms)")

    # Show alerts
    alerts = monitor.get_alerts()
    if alerts:
        print("\nAlerts:")
        for alert in alerts:
            print(f"  {alert}")


def demonstrate_circuit_breaker():
    """Demonstrate circuit breaker"""
    print("\n=== CIRCUIT BREAKER ===")

    circuit = CircuitBreaker(failure_threshold=3, timeout=2.0)

    # Simulate unreliable service
    call_count = 0
    def unreliable_service():
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            raise RuntimeError("Service error")
        return "Success"

    # Make calls that will fail
    print("\nCalling unreliable service:")
    for i in range(3):
        try:
            result = circuit.call(unreliable_service)
            print(f"  Call {i+1}: {result}")
        except Exception as e:
            print(f"  Call {i+1}: Failed - {e}")

    print(f"Circuit state: {circuit.get_state().value}")

    # Circuit is now open - requests blocked
    print("\nCircuit is OPEN, requests blocked:")
    try:
        circuit.call(unreliable_service)
    except RuntimeError as e:
        print(f"  {e}")

    # Wait for timeout
    print("\nWaiting for circuit timeout...")
    time.sleep(2.1)

    # Try again - the circuit should be half-open
    print("\nRetrying after timeout:")
    try:
        result = circuit.call(unreliable_service)
        print(f"  Call succeeded: {result}")
        print(f"  Circuit state: {circuit.get_state().value}")
    except Exception as e:
        print(f"  Call failed: {e}")


def main():
    """Run all demonstrations"""
    print("AVAILABILITY PATTERNS IN DISTRIBUTED SYSTEMS")
    print("=" * 60)

    demonstrate_redundancy()
    demonstrate_failover()
    demonstrate_replication()
    demonstrate_health_monitoring()
    demonstrate_circuit_breaker()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("1. Redundancy - Eliminate single points of failure")
    print("2. Failover - Automatic recovery from failures")
    print("3. Replication - Data durability across multiple nodes")
    print("4. Health monitoring - Detect failures early")
    print("5. Circuit breaker - Prevent cascading failures")
    print("\nHigh availability = Redundancy + Fast failure detection + Automatic recovery")


if __name__ == "__main__":
    main()