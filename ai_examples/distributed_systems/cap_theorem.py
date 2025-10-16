"""
CAP Theorem in Distributed Systems

CAP Theorem states that a distributed system can only guarantee TWO of the following three properties:
- Consistency (C): All nodes see the same data at the same time
- Availability (A): Every request receives a response (success or failure)
- Partition Tolerance (P): System continues to operate despite network partitions

Since network partitions are inevitable in distributed systems, you must choose between:
- CP (Consistency and Partition Tolerance): Sacrifice availability during partitions
- AP (Availability and Partition Tolerance): Sacrifice consistency during partitions

Key concepts:
1. CP Systems - Prioritize consistency (e.g., traditional databases)
2. AP Systems - Prioritize availability (e.g., Cassandra, DynamoDB)
3. Network Partitions - When nodes can't communicate
4. Eventual Consistency - AP systems converge to consistency over time
5. Quorum Consensus - Balance between CP and AP
"""

import time
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum


# 1. CP SYSTEM (Consistency + Partition Tolerance)
class NodeState(Enum):
    """State of a node in the cluster"""
    LEADER = "leader"
    FOLLOWER = "follower"
    PARTITIONED = "partitioned"


@dataclass
class CPNode:
    """Node in a CP (Consistent + Partition Tolerant) system"""
    node_id: str
    state: NodeState = NodeState.FOLLOWER
    data: dict = field(default_factory=dict)
    is_reachable: bool = True
    version: int = 0

    def can_serve_reads(self) -> bool:
        """CP systems only serve reads when connected to a leader"""
        return self.is_reachable and self.state in [NodeState.LEADER, NodeState.FOLLOWER]

    def can_serve_writes(self) -> bool:
        """Only leader can serve writes in CP systems"""
        return self.is_reachable and self.state == NodeState.LEADER


class CPSystem:
    """
    CP System: Prioritizes Consistency and Partition Tolerance.
    Sacrifices availability during network partitions to maintain consistency.
    Example: ZooKeeper, etcd, traditional RDBMS with strong consistency.
    """

    def __init__(self, num_nodes: int = 3):
        self.nodes = [CPNode(node_id=f"node_{i}") for i in range(num_nodes)]
        self.leader_id = 0
        self.nodes[self.leader_id].state = NodeState.LEADER
        self.write_count = 0
        self.read_count = 0
        self.rejected_operations = 0

    def _get_leader(self) -> Optional[CPNode]:
        """Get the current leader node"""
        return self.nodes[self.leader_id] if self.leader_id < len(self.nodes) else None

    def _replicate_to_followers(self, key: str, value: Any) -> bool:
        """Replicate data to all reachable followers"""
        leader = self._get_leader()
        if not leader:
            return False

        # Require a majority of nodes to acknowledge (strong consistency)
        acks = 1  # Leader counts as one ack
        required_acks = (len(self.nodes) // 2) + 1

        for node in self.nodes:
            if node.node_id != leader.node_id and node.is_reachable:
                node.data[key] = value
                node.version = leader.version
                acks += 1

        return acks >= required_acks

    def write(self, key: str, value: Any) -> dict:
        """Write data with strong consistency"""
        leader = self._get_leader()

        if not leader or not leader.can_serve_writes():
            self.rejected_operations += 1
            return {
                "success": False,
                "error": "No leader available - system unavailable during partition",
                "consistency": "maintained"
            }

        # Write to the leader
        leader.data[key] = value
        leader.version += 1

        # Replicate to followers (require majority)
        if self._replicate_to_followers(key, value):
            self.write_count += 1
            return {
                "success": True,
                "node": leader.node_id,
                "version": leader.version,
                "consistency": "strong"
            }
        else:
            # Rollback if it can't achieve majority
            del leader.data[key]
            self.rejected_operations += 1
            return {
                "success": False,
                "error": "Cannot achieve majority - rejecting write",
                "consistency": "maintained"
            }

    def read(self, key: str) -> dict:
        """Read data with strong consistency"""
        leader = self._get_leader()

        if not leader or not leader.can_serve_reads():
            self.rejected_operations += 1
            return {
                "success": False,
                "error": "System unavailable during partition",
                "consistency": "maintained"
            }

        self.read_count += 1
        return {
            "success": True,
            "value": leader.data.get(key),
            "node": leader.node_id,
            "version": leader.version,
            "consistency": "strong"
        }

    def simulate_partition(self, node_ids: list[int]):
        """Simulate network partition"""
        for node_id in node_ids:
            if node_id < len(self.nodes):
                self.nodes[node_id].is_reachable = False
                self.nodes[node_id].state = NodeState.PARTITIONED
        print(f"  [CP] Partitioned nodes: {node_ids} - System may become unavailable")

    def heal_partition(self):
        """Heal network partition"""
        for node in self.nodes:
            node.is_reachable = True
            if node.state == NodeState.PARTITIONED:
                node.state = NodeState.FOLLOWER
        print(f"  [CP] Partition healed - System available again")

    def get_stats(self) -> dict:
        """Get system statistics"""
        reachable_nodes = sum(1 for n in self.nodes if n.is_reachable)
        return {
            "type": "CP (Consistency + Partition Tolerance)",
            "write_count": self.write_count,
            "read_count": self.read_count,
            "rejected_operations": self.rejected_operations,
            "reachable_nodes": f"{reachable_nodes}/{len(self.nodes)}",
            "guarantee": "Strong consistency, may sacrifice availability"
        }


# 2. AP SYSTEM (Availability + Partition Tolerance)
@dataclass
class APNode:
    """Node in an AP (Available + Partition Tolerant) system"""
    node_id: str
    data: dict = field(default_factory=dict)
    vector_clock: dict = field(default_factory=dict)  # For conflict detection
    is_reachable: bool = True

    def update_vector_clock(self, key: str):
        """Update vector clock for conflict resolution"""
        if key not in self.vector_clock:
            self.vector_clock[key] = 0
        self.vector_clock[key] += 1


class APSystem:
    """
    AP System: Prioritizes Availability and Partition Tolerance.
    Sacrifices strong consistency for availability (eventual consistency).
    Example: Cassandra, DynamoDB, Riak.
    """

    def __init__(self, num_nodes: int = 3, replication_factor: int = 2):
        self.nodes = [APNode(node_id=f"node_{i}") for i in range(num_nodes)]
        self.replication_factor = min(replication_factor, num_nodes)
        self.write_count = 0
        self.read_count = 0
        self.conflicts_detected = 0

    def _get_nodes_for_key(self, key: str) -> list[APNode]:
        """Get nodes responsible for a key (consistent hashing simulation)"""
        hash_val = hash(key)
        start_idx = hash_val % len(self.nodes)

        # Return replication_factor nodes
        selected = []
        for i in range(self.replication_factor):
            idx = (start_idx + i) % len(self.nodes)
            selected.append(self.nodes[idx])
        return selected

    def _get_reachable_nodes(self, nodes: list[APNode]) -> list[APNode]:
        """Get reachable nodes from a list"""
        return [n for n in nodes if n.is_reachable]

    def write(self, key: str, value: Any) -> dict:
        """Write data with eventual consistency"""
        target_nodes = self._get_nodes_for_key(key)
        reachable_nodes = self._get_reachable_nodes(target_nodes)

        if not reachable_nodes:
            return {
                "success": False,
                "error": "No reachable nodes (extremely rare)",
                "consistency": "eventual"
            }

        # Write to any available node (W=1 for maximum availability)
        written_to = []
        for node in reachable_nodes:
            node.data[key] = value
            node.update_vector_clock(key)
            written_to.append(node.node_id)

        self.write_count += 1
        return {
            "success": True,
            "nodes": written_to,
            "consistency": "eventual",
            "note": "Data will be replicated asynchronously"
        }

    def read(self, key: str) -> dict:
        """Read data with eventual consistency"""
        target_nodes = self._get_nodes_for_key(key)
        reachable_nodes = self._get_reachable_nodes(target_nodes)

        if not reachable_nodes:
            return {
                "success": False,
                "error": "No reachable nodes",
                "consistency": "eventual"
            }

        # Read from any available node (R=1 for maximum availability)
        node = reachable_nodes[0]
        value = node.data.get(key)

        # Check for conflicts across reachable nodes
        values = [n.data.get(key) for n in reachable_nodes]
        has_conflict = len(set(str(v) for v in values)) > 1

        if has_conflict:
            self.conflicts_detected += 1

        self.read_count += 1
        return {
            "success": True,
            "value": value,
            "node": node.node_id,
            "consistency": "eventual",
            "conflict_detected": has_conflict,
            "note": "May return stale data during partition"
        }

    def simulate_partition(self, node_ids: list[int]):
        """Simulate network partition"""
        for node_id in node_ids:
            if node_id < len(self.nodes):
                self.nodes[node_id].is_reachable = False
        print(f"  [AP] Partitioned nodes: {node_ids} - System remains available")

    def heal_partition(self):
        """Heal network partition and trigger reconciliation"""
        for node in self.nodes:
            node.is_reachable = True
        print(f"  [AP] Partition healed - Reconciling data (eventual consistency)")
        self._reconcile_data()

    def _reconcile_data(self):
        """Reconcile data across nodes after partition heals"""
        # Simulate data reconciliation using vector clocks
        all_keys = set()
        for node in self.nodes:
            all_keys.update(node.data.keys())

        for key in all_keys:
            # Get the latest value based on a vector clock
            latest_version = 0
            latest_value = None

            for node in self.nodes:
                if key in node.vector_clock and node.vector_clock[key] > latest_version:
                    latest_version = node.vector_clock[key]
                    latest_value = node.data.get(key)

            # Propagate latest value to all nodes
            for node in self.nodes:
                if latest_value is not None:
                    node.data[key] = latest_value
                    node.vector_clock[key] = latest_version

    def get_stats(self) -> dict:
        """Get system statistics"""
        reachable_nodes = sum(1 for n in self.nodes if n.is_reachable)
        return {
            "type": "AP (Availability + Partition Tolerance)",
            "write_count": self.write_count,
            "read_count": self.read_count,
            "conflicts_detected": self.conflicts_detected,
            "reachable_nodes": f"{reachable_nodes}/{len(self.nodes)}",
            "guarantee": "Always available, eventual consistency"
        }


# 3. QUORUM CONSENSUS (Balance between CP and AP)
@dataclass
class QuorumNode:
    """Node in a quorum-based system"""
    node_id: str
    data: dict = field(default_factory=dict)
    version: dict = field(default_factory=dict)  # Key -> version
    is_reachable: bool = True


class QuorumSystem:
    """
    Quorum-based system: Configurable consistency/availability trade-off.
    Uses R (read quorum) and W (write quorum) to balance between CP and AP.

    Rules:
    - W + R > N: Strong consistency (reads see latest writes)
    - W + R <= N: Eventual consistency (may read stale data)
    - W = N: Strongest consistency, lowest availability
    - W = 1: Highest availability, the weakest consistency
    """

    def __init__(self, num_nodes: int = 5, write_quorum: int = 3, read_quorum: int = 3):
        self.nodes = [QuorumNode(node_id=f"node_{i}") for i in range(num_nodes)]
        self.write_quorum = write_quorum
        self.read_quorum = read_quorum
        self.write_count = 0
        self.read_count = 0
        self.quorum_failures = 0

    def _get_reachable_nodes(self) -> list[QuorumNode]:
        """Get all reachable nodes"""
        return [n for n in self.nodes if n.is_reachable]

    def write(self, key: str, value: Any) -> dict:
        """Write data with quorum consensus"""
        reachable = self._get_reachable_nodes()

        if len(reachable) < self.write_quorum:
            self.quorum_failures += 1
            return {
                "success": False,
                "error": f"Cannot achieve write quorum (need {self.write_quorum}, have {len(reachable)})",
                "quorum": f"W={self.write_quorum}"
            }

        # Write to quorum nodes
        version = int(time.time() * 1000)  # Timestamp as version
        written_to = []

        for node in reachable[:self.write_quorum]:
            node.data[key] = value
            node.version[key] = version
            written_to.append(node.node_id)

        self.write_count += 1
        return {
            "success": True,
            "nodes": written_to,
            "version": version,
            "quorum": f"W={self.write_quorum}/{len(self.nodes)}"
        }

    def read(self, key: str) -> dict:
        """Read data with quorum consensus"""
        reachable = self._get_reachable_nodes()

        if len(reachable) < self.read_quorum:
            self.quorum_failures += 1
            return {
                "success": False,
                "error": f"Cannot achieve read quorum (need {self.read_quorum}, have {len(reachable)})",
                "quorum": f"R={self.read_quorum}"
            }

        # Read from quorum nodes and get the latest version
        latest_version = 0
        latest_value = None

        for node in reachable[:self.read_quorum]:
            if key in node.version and node.version[key] > latest_version:
                latest_version = node.version[key]
                latest_value = node.data.get(key)

        self.read_count += 1
        return {
            "success": True,
            "value": latest_value,
            "version": latest_version,
            "quorum": f"R={self.read_quorum}/{len(self.nodes)}"
        }

    def simulate_partition(self, node_ids: list[int]):
        """Simulate network partition"""
        for node_id in node_ids:
            if node_id < len(self.nodes):
                self.nodes[node_id].is_reachable = False
        reachable = len(self._get_reachable_nodes())
        can_write = reachable >= self.write_quorum
        can_read = reachable >= self.read_quorum
        print(f"  [QUORUM] Partitioned nodes: {node_ids}")
        print(f"    Can write: {can_write}, Can read: {can_read}")

    def heal_partition(self):
        """Heal network partition"""
        for node in self.nodes:
            node.is_reachable = True
        print(f"  [QUORUM] Partition healed")

    def get_stats(self) -> dict:
        """Get system statistics"""
        reachable_nodes = len(self._get_reachable_nodes())
        consistency_level = "Strong" if (self.write_quorum + self.read_quorum) > len(self.nodes) else "Eventual"

        return {
            "type": "Quorum-based System",
            "write_count": self.write_count,
            "read_count": self.read_count,
            "quorum_failures": self.quorum_failures,
            "reachable_nodes": f"{reachable_nodes}/{len(self.nodes)}",
            "config": f"W={self.write_quorum}, R={self.read_quorum}, N={len(self.nodes)}",
            "consistency": consistency_level
        }


def demonstrate_cp_system():
    """Demonstrate CP system (Consistency and Partition Tolerance)"""
    print("\n=== CP SYSTEM (Consistency + Partition Tolerance) ===")
    print("Guarantees: Strong consistency, sacrifices availability during partitions")
    print("Examples: ZooKeeper, etcd, traditional RDBMS\n")

    cp_system = CPSystem(num_nodes=3)

    # Normal operation
    print("Normal operation:")
    result = cp_system.write("config:timeout", 30)
    print(f"  Write: {result}")
    result = cp_system.read("config:timeout")
    print(f"  Read: {result}")

    # Simulate partition (lose majority)
    print("\nSimulating network partition (2/3 nodes unreachable):")
    cp_system.simulate_partition([1, 2])

    # Try operations during partition
    print("Attempting operations during partition:")
    result = cp_system.write("config:retry", 3)
    print(f"  Write: {result}")
    result = cp_system.read("config:timeout")
    print(f"  Read: {result}")

    # Heal partition
    print("\nHealing partition:")
    cp_system.heal_partition()
    result = cp_system.write("config:retry", 3)
    print(f"  Write after heal: {result}")

    print(f"\nStats: {cp_system.get_stats()}")


def demonstrate_ap_system():
    """Demonstrate AP system (Availability and Partition Tolerance)"""
    print("\n=== AP SYSTEM (Availability + Partition Tolerance) ===")
    print("Guarantees: Always available, eventual consistency")
    print("Examples: Cassandra, DynamoDB, Riak\n")

    ap_system = APSystem(num_nodes=3, replication_factor=2)

    # Normal operation
    print("Normal operation:")
    result = ap_system.write("user:1", {"name": "Alice", "status": "active"})
    print(f"  Write: {result}")
    result = ap_system.read("user:1")
    print(f"  Read: {result}")

    # Simulate partition
    print("\nSimulating network partition (1/3 nodes unreachable):")
    ap_system.simulate_partition([2])

    # Operations still work during partition
    print("Operations during partition (still available!):")
    result = ap_system.write("user:2", {"name": "Bob", "status": "active"})
    print(f"  Write: {result}")
    result = ap_system.read("user:1")
    print(f"  Read: {result}")

    # Heal partition
    print("\nHealing partition:")
    ap_system.heal_partition()

    print(f"\nStats: {ap_system.get_stats()}")


def demonstrate_quorum_system():
    """Demonstrate a Quorum-based system"""
    print("\n=== QUORUM SYSTEM (Configurable Trade-off) ===")
    print("Balance between CP and AP using W (write) and R (read) quorums\n")

    # Strong consistency: W=3, R=3, N=5 (W+R > N)
    print("Configuration: W=3, R=3, N=5 (Strong consistency)")
    quorum = QuorumSystem(num_nodes=5, write_quorum=3, read_quorum=3)

    # Normal operation
    print("\nNormal operation:")
    result = quorum.write("key1", "value1")
    print(f"  Write: {result}")
    result = quorum.read("key1")
    print(f"  Read: {result}")

    # Simulate partition (lose 2 nodes)
    print("\nSimulating partition (2/5 nodes unreachable):")
    quorum.simulate_partition([3, 4])

    # Can still achieve quorum
    print("Operations during partition:")
    result = quorum.write("key2", "value2")
    print(f"  Write: {result}")
    result = quorum.read("key1")
    print(f"  Read: {result}")

    # Lose one more node (can't achieve quorum)
    print("\nLosing another node (3/5 unreachable, can't achieve quorum):")
    quorum.simulate_partition([2])

    result = quorum.write("key3", "value3")
    print(f"  Write: {result}")

    print(f"\nStats: {quorum.get_stats()}")


def demonstrate_cap_tradeoffs():
    """Demonstrate CAP theorem trade-offs"""
    print("\n=== CAP THEOREM TRADE-OFFS ===")
    print("\nCAP Theorem: You can only have TWO of:")
    print("  C - Consistency: All nodes see the same data")
    print("  A - Availability: Every request gets a response")
    print("  P - Partition Tolerance: System works despite network failures")
    print("\nSince network partitions are inevitable, choose between:")
    print("  CP: Consistency + Partition Tolerance (sacrifice availability)")
    print("  AP: Availability + Partition Tolerance (sacrifice consistency)")

    print("\n" + "=" * 60)
    print("COMPARISON DURING NETWORK PARTITION:")
    print("=" * 60)

    comparison = [
        {
            "System": "CP (ZooKeeper)",
            "Consistency": "Strong",
            "Availability": "May be unavailable",
            "Use Case": "Configuration, coordination"
        },
        {
            "System": "AP (Cassandra)",
            "Consistency": "Eventual",
            "Availability": "Always available",
            "Use Case": "User profiles, time-series"
        },
        {
            "System": "Quorum (W=3,R=3)",
            "Consistency": "Tunable",
            "Availability": "Tunable",
            "Use Case": "Configurable requirements"
        }
    ]

    for system in comparison:
        print(f"\n{system['System']}:")
        print(f"  Consistency: {system['Consistency']}")
        print(f"  Availability: {system['Availability']}")
        print(f"  Use Case: {system['Use Case']}")


def main():
    """Run all demonstrations"""
    print("CAP THEOREM IN DISTRIBUTED SYSTEMS")
    print("=" * 60)

    demonstrate_cp_system()
    demonstrate_ap_system()
    demonstrate_quorum_system()
    demonstrate_cap_tradeoffs()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("1. CP systems: Strong consistency, may sacrifice availability")
    print("2. AP systems: Always available, eventual consistency")
    print("3. Quorum systems: Configurable trade-off using W/R parameters")
    print("4. Network partitions are inevitable in distributed systems")
    print("5. Choose based on your requirements: consistency vs availability")
    print("\nCAP Theorem: In the presence of partitions, choose C or A (not both)")


if __name__ == "__main__":
    main()
