"""
Single Point of Failure (SPOF) Pattern

Demonstrates the antipattern of having a single point of failure and how to mitigate it
with redundancy, failover mechanisms, and distributed architectures.
"""

import random
from abc import ABC, abstractmethod
from typing import List


# ANTI-PATTERN: Single Point of Failure
class SingleDatabase:
    """A single database instance - if it fails, the entire system fails"""

    def __init__(self):
        self.is_healthy = True

    def query(self, sql: str):
        if not self.is_healthy:
            raise RuntimeError("Database is down!")
        return f"Result of: {sql}"


class ApplicationWithSPOF:
    """Application that relies on a single database"""

    def __init__(self):
        self.database = SingleDatabase()

    def get_data(self):
        try:
            return self.database.query("SELECT * FROM users")
        except Exception as e:
            # If database fails, application fails
            raise RuntimeError(f"Application failed: {e}")


# SOLUTION: Redundancy and Failover
class DatabaseNode(ABC):
    """Abstract database node"""

    def __init__(self, name: str):
        self.name = name
        self.is_healthy = True

    @abstractmethod
    def query(self, sql: str) -> str:
        pass

    def health_check(self) -> bool:
        return self.is_healthy


class PrimaryDatabase(DatabaseNode):
    """Primary database node"""

    def query(self, sql: str) -> str:
        if not self.is_healthy:
            raise RuntimeError(f"{self.name} is down!")
        return f"[PRIMARY] Result of: {sql}"


class ReplicaDatabase(DatabaseNode):
    """Replica database node"""

    def query(self, sql: str) -> str:
        if not self.is_healthy:
            raise RuntimeError(f"{self.name} is down!")
        return f"[REPLICA] Result of: {sql}"


class DatabaseCluster:
    """Database cluster with automatic failover"""

    def __init__(self, primary: DatabaseNode, replicas: List[DatabaseNode]):
        self.primary = primary
        self.replicas = replicas
        self.all_nodes = [primary] + replicas

    def query(self, sql: str) -> str:
        # Try primary first
        if self.primary.health_check():
            try:
                return self.primary.query(sql)
            except RuntimeError:
                pass

        # Failover to replicas
        for replica in self.replicas:
            if replica.health_check():
                try:
                    return replica.query(sql)
                except RuntimeError:
                    continue

        # All nodes failed
        raise RuntimeError("All database nodes are down!")


class ResilientApplication:
    """Application with redundant database connections"""

    def __init__(self, database_cluster: DatabaseCluster):
        self.database_cluster = database_cluster

    def get_data(self):
        try:
            return self.database_cluster.query("SELECT * FROM users")
        except RuntimeError as e:
            # Gracefully handle complete failure
            return f"Service degraded: {e}"


# Load Balancer Pattern (distributes load across multiple instances)
class ServiceInstance:
    """Individual service instance"""

    def __init__(self, instance_id: int):
        self.instance_id = instance_id
        self.is_healthy = True

    def handle_request(self, request: str) -> str:
        if not self.is_healthy:
            raise RuntimeError(f"Instance {self.instance_id} is down!")
        return f"Instance {self.instance_id} processed: {request}"


class LoadBalancer:
    """Load balancer distributes requests across healthy instances"""

    def __init__(self, instances: List[ServiceInstance]):
        self.instances = instances

    def route_request(self, request: str) -> str:
        # Get healthy instances
        healthy_instances = [i for i in self.instances if i.is_healthy]

        if not healthy_instances:
            raise RuntimeError("No healthy instances available!")

        # Round-robin or random selection
        instance = random.choice(healthy_instances)
        return instance.handle_request(request)


# Example usage
if __name__ == "__main__":
    print("=== ANTI-PATTERN: Single Point of Failure ===")
    app_spof = ApplicationWithSPOF()
    print(app_spof.get_data())

    # Simulate failure
    app_spof.database.is_healthy = False
    try:
        app_spof.get_data()
    except RuntimeError as exp:
        print(f"FAILURE: {exp}\n")

    print("=== SOLUTION: Database Cluster with Failover ===")
    db_primary = PrimaryDatabase("primary-db")
    db_replica_1 = ReplicaDatabase("replica-1")
    db_replica_2 = ReplicaDatabase("replica-2")

    cluster = DatabaseCluster(db_primary, [db_replica_1, db_replica_2])
    resilient_app = ResilientApplication(cluster)

    print(resilient_app.get_data())

    # Simulate primary failure
    print("\nSimulating primary database failure...")
    db_primary.is_healthy = False
    print(resilient_app.get_data())

    # Simulate multiple failures
    print("\nSimulating replica-1 failure...")
    db_replica_1.is_healthy = False
    print(resilient_app.get_data())

    # Simulate complete failure
    print("\nSimulating complete cluster failure...")
    db_replica_2.is_healthy = False
    print(resilient_app.get_data())

    print("\n=== SOLUTION: Load Balancer ===")
    services = [ServiceInstance(i) for i in range(1, 4)]
    load_balancer = LoadBalancer(services)

    for i in range(5):
        print(load_balancer.route_request(f"request-{i}"))

    # Simulate instance failure
    print("\nSimulating instance 2 failure...")
    services[1].is_healthy = False
    for i in range(5, 8):
        print(load_balancer.route_request(f"request-{i}"))