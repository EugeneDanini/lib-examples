"""
Scalability Patterns in Distributed Systems

Scalability refers to a system's ability to handle an increased load by adding resources.
Two main types:
- Horizontal Scaling (scale out): Add more machines/instances
- Vertical Scaling (scale up): Add more resources to existing machines

Key scalability patterns:
1. Load Balancing - Distribute work across multiple instances
2. Sharding/Partitioning - Split data across multiple databases
3. Caching - Reduce a load by storing frequently accessed data
4. Async Processing - Offload work to background queues
5. Database Read Replicas - Scale read operations
"""

import time
import random
from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib


# 1. LOAD BALANCING
class LoadBalancerStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    WEIGHTED = "weighted"


@dataclass
class Server:
    """Server instance"""
    id: str
    host: str
    port: int
    active_connections: int = 0
    weight: int = 1
    is_healthy: bool = True

    def handle_request(self, request: str) -> str:
        """Handle a request"""
        self.active_connections += 1
        # Simulate request processing
        time.sleep(0.001)
        result = f"Server {self.id} processed: {request}"
        self.active_connections -= 1
        return result


class LoadBalancer:
    """
    Load balancer that distributes requests across multiple servers
    using different strategies.
    """

    def __init__(self, strategy: LoadBalancerStrategy = LoadBalancerStrategy.ROUND_ROBIN):
        self.servers: list[Server] = []
        self.strategy = strategy
        self.current_index = 0

    def add_server(self, server: Server):
        """Add a server to the pool"""
        self.servers.append(server)
        print(f"Added server: {server.id}")

    def remove_server(self, server_id: str):
        """Remove a server from the pool"""
        self.servers = [s for s in self.servers if s.id != server_id]
        print(f"Removed server: {server_id}")

    def get_healthy_servers(self) -> list[Server]:
        """Get the list of healthy servers"""
        return [s for s in self.servers if s.is_healthy]

    def select_server(self) -> Optional[Server]:
        """Select a server based on the load balancing strategy"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None

        if self.strategy == LoadBalancerStrategy.ROUND_ROBIN:
            return self._round_robin(healthy_servers)
        elif self.strategy == LoadBalancerStrategy.LEAST_CONNECTIONS:
            return self._least_connections(healthy_servers)
        elif self.strategy == LoadBalancerStrategy.RANDOM:
            return random.choice(healthy_servers)
        elif self.strategy == LoadBalancerStrategy.WEIGHTED:
            return self._weighted(healthy_servers)
        return None

    def _round_robin(self, servers: list[Server]) -> Server:
        """Round-robin selection"""
        server = servers[self.current_index % len(servers)]
        self.current_index += 1
        return server

    def _least_connections(self, servers: list[Server]) -> Server:
        """Select a server with the least active connections"""
        return min(servers, key=lambda s: s.active_connections)

    def _weighted(self, servers: list[Server]) -> Server:
        """Weighted random selection"""
        total_weight = sum(s.weight for s in servers)
        rand_val = random.randint(1, total_weight)
        cumulative = 0
        for server in servers:
            cumulative += server.weight
            if rand_val <= cumulative:
                return server
        return servers[-1]

    def handle_request(self, request: str) -> str:
        """Handle a request by routing to a server"""
        server = self.select_server()
        if not server:
            return "Error: No healthy servers available"
        return server.handle_request(request)

    def get_stats(self) -> dict:
        """Get load balancer statistics"""
        return {
            "total_servers": len(self.servers),
            "healthy_servers": len(self.get_healthy_servers()),
            "connections_per_server": {
                s.id: s.active_connections for s in self.servers
            }
        }


# 2. SHARDING/PARTITIONING
class ShardStrategy(Enum):
    """Sharding strategies"""
    HASH = "hash"
    RANGE = "range"
    GEOGRAPHIC = "geographic"


class DatabaseShard:
    """A single database shard"""

    def __init__(self, shard_id: int, name: str):
        self.shard_id = shard_id
        self.name = name
        self.data = {}

    def write(self, key: str, value: Any):
        """Write data to this shard"""
        self.data[key] = value

    def read(self, key: str) -> Optional[Any]:
        """Read data from this shard"""
        return self.data.get(key)

    def get_size(self) -> int:
        """Get the number of records in this shard"""
        return len(self.data)


class ShardedDatabase:
    """
    Database that partitions data across multiple shards
    for horizontal scalability.
    """

    def __init__(self, num_shards: int = 4, strategy: ShardStrategy = ShardStrategy.HASH):
        self.shards = [
            DatabaseShard(i, f"shard_{i}")
            for i in range(num_shards)
        ]
        self.strategy = strategy

    def _get_shard_by_hash(self, key: str) -> DatabaseShard:
        """Get shard using consistent hashing"""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        shard_index = hash_value % len(self.shards)
        return self.shards[shard_index]

    def _get_shard_by_range(self, key: str) -> DatabaseShard:
        """Get shard using range-based partitioning"""
        # Simple range: a-f -> shard 0, g-m -> shard 1, etc.
        first_char = key[0].lower() if key else 'a'
        range_size = 26 // len(self.shards)
        shard_index = min((ord(first_char) - ord('a')) // range_size, len(self.shards) - 1)
        return self.shards[shard_index]

    def get_shard(self, key: str) -> DatabaseShard:
        """Get the appropriate shard for a key"""
        if self.strategy == ShardStrategy.HASH:
            return self._get_shard_by_hash(key)
        elif self.strategy == ShardStrategy.RANGE:
            return self._get_shard_by_range(key)
        else:
            return self.shards[0]  # Default fallback

    def write(self, key: str, value: Any):
        """Write data to the appropriate shard"""
        shard = self.get_shard(key)
        shard.write(key, value)

    def read(self, key: str) -> Optional[Any]:
        """Read data from the appropriate shard"""
        shard = self.get_shard(key)
        return shard.read(key)

    def get_stats(self) -> dict:
        """Get sharding statistics"""
        return {
            "total_shards": len(self.shards),
            "records_per_shard": {
                shard.name: shard.get_size()
                for shard in self.shards
            },
            "total_records": sum(s.get_size() for s in self.shards)
        }


# 3. CACHING
class CacheEvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    access_count: int = 0
    last_access_time: float = 0.0


class Cache:
    """
    In-memory cache with different eviction policies.
    Reduces the load on backend systems by storing frequently accessed data.
    """

    def __init__(self, max_size: int = 100, policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU):
        self.max_size = max_size
        self.policy = policy
        self.cache: dict[str, CacheEntry] = {}
        self.access_order: list[str] = []  # For LRU/FIFO
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from a cache"""
        if key in self.cache:
            self.hits += 1
            entry = self.cache[key]
            entry.access_count += 1
            entry.last_access_time = time.time()

            # Update access order for LRU
            if self.policy == CacheEvictionPolicy.LRU:
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)

            return entry.value
        else:
            self.misses += 1
            return None

    def put(self, key: str, value: Any):
        """Put value in a cache"""
        # If the cache is full, evict based on policy
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict()

        # Add or update entry
        if key not in self.cache:
            self.cache[key] = CacheEntry(
                key=key,
                value=value,
                access_count=1,
                last_access_time=time.time()
            )
            if self.policy in [CacheEvictionPolicy.LRU, CacheEvictionPolicy.FIFO]:
                self.access_order.append(key)
        else:
            self.cache[key].value = value
            self.cache[key].access_count += 1
            self.cache[key].last_access_time = time.time()

    def _evict(self):
        """Evict an entry based on policy"""
        if not self.cache:
            return

        if self.policy == CacheEvictionPolicy.LRU:
            # Remove least recently used
            evict_key = self.access_order.pop(0)
        elif self.policy == CacheEvictionPolicy.FIFO:
            # Remove first in
            evict_key = self.access_order.pop(0)
        elif self.policy == CacheEvictionPolicy.LFU:
            # Remove least frequently used
            evict_key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
        else:
            evict_key = list(self.cache.keys())[0]

        del self.cache[evict_key]

    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }


# 4. ASYNC PROCESSING
class TaskStatus(Enum):
    """Task processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Background task"""
    id: str
    type: str
    payload: dict
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    created_at: float = 0.0


class TaskQueue:
    """
    Asynchronous task queue for offloading work to background processors.
    Improves scalability by handling requests quickly and processing work later.
    """

    def __init__(self):
        self.queue: list[Task] = []
        self.completed: dict[str, Task] = {}
        self.task_counter = 0

    def enqueue(self, task_type: str, payload: dict) -> str:
        """Add a task to the queue"""
        self.task_counter += 1
        task = Task(
            id=f"task_{self.task_counter}",
            type=task_type,
            payload=payload,
            created_at=time.time()
        )
        self.queue.append(task)
        return task.id

    def dequeue(self) -> Optional[Task]:
        """Get next task from queue"""
        if self.queue:
            return self.queue.pop(0)
        return None

    def process_task(self, task: Task):
        """Process a task (simulated)"""
        task.status = TaskStatus.PROCESSING

        try:
            # Simulate task processing
            if task.type == "send_email":
                time.sleep(0.01)
                task.result = f"Email sent to {task.payload.get('email')}"
            elif task.type == "generate_report":
                time.sleep(0.02)
                task.result = f"Report generated: {task.payload.get('report_name')}"
            elif task.type == "process_image":
                time.sleep(0.015)
                task.result = f"Image processed: {task.payload.get('image_id')}"
            else:
                task.result = f"Processed {task.type}"

            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = str(e)

        self.completed[task.id] = task

    def process_batch(self, batch_size: int = 10):
        """Process a batch of tasks"""
        processed = 0
        while processed < batch_size and self.queue:
            task = self.dequeue()
            if task:
                self.process_task(task)
                processed += 1
        return processed

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a specific task"""
        # Check completed tasks
        if task_id in self.completed:
            return self.completed[task_id].status

        # Check pending tasks
        for task in self.queue:
            if task.id == task_id:
                return task.status

        return None

    def get_stats(self) -> dict:
        """Get queue statistics"""
        return {
            "pending": len(self.queue),
            "completed": len([t for t in self.completed.values() if t.status == TaskStatus.COMPLETED]),
            "failed": len([t for t in self.completed.values() if t.status == TaskStatus.FAILED]),
            "total_processed": len(self.completed)
        }


# 5. DATABASE READ REPLICAS
class DatabaseNode:
    """A database node (master or replica)"""

    def __init__(self, node_id: str, is_master: bool = False):
        self.node_id = node_id
        self.is_master = is_master
        self.data = {}
        self.read_count = 0
        self.write_count = 0

    def read(self, key: str) -> Optional[Any]:
        """Read from a database"""
        self.read_count += 1
        return self.data.get(key)

    def write(self, key: str, value: Any):
        """Write to a database"""
        if not self.is_master:
            raise PermissionError("Cannot write to read replica")
        self.write_count += 1
        self.data[key] = value


class ReplicatedDatabase:
    """
    Database with read replicas to scale read operations.
    Writes go to master, reads distributed across replicas.
    """

    def __init__(self, num_replicas: int = 3):
        self.master = DatabaseNode("master", is_master=True)
        self.replicas = [
            DatabaseNode(f"replica_{i}")
            for i in range(num_replicas)
        ]
        self.current_replica_index = 0

    def write(self, key: str, value: Any):
        """Write to master and replicate to all replicas"""
        # Write to master
        self.master.write(key, value)

        # Replicate to all replicas (simulate replication)
        for replica in self.replicas:
            replica.data[key] = value

    def read(self, key: str) -> Optional[Any]:
        """Read from a replica (load balanced)"""
        # Round-robin across replicas
        replica = self.replicas[self.current_replica_index % len(self.replicas)]
        self.current_replica_index += 1
        return replica.read(key)

    def get_stats(self) -> dict:
        """Get replication statistics"""
        return {
            "master_writes": self.master.write_count,
            "master_reads": self.master.read_count,
            "replica_reads": {
                r.node_id: r.read_count
                for r in self.replicas
            },
            "total_replica_reads": sum(r.read_count for r in self.replicas)
        }


def demonstrate_load_balancing():
    """Demonstrate load balancing"""
    print("\n=== LOAD BALANCING ===")

    # Create load balancer with round-robin strategy
    lb = LoadBalancer(strategy=LoadBalancerStrategy.ROUND_ROBIN)

    # Add servers
    lb.add_server(Server("server1", "192.168.1.1", 8080, weight=2))
    lb.add_server(Server("server2", "192.168.1.2", 8080, weight=1))
    lb.add_server(Server("server3", "192.168.1.3", 8080, weight=1))

    # Process requests
    print("\nProcessing 6 requests with round-robin:")
    for i in range(6):
        result = lb.handle_request(f"request_{i}")
        print(f"  {result}")

    # Try the least connection strategy
    lb.strategy = LoadBalancerStrategy.LEAST_CONNECTIONS
    print("\nSwitched to least connections strategy")

    stats = lb.get_stats()
    print(f"Stats: {stats}")


def demonstrate_sharding():
    """Demonstrate database sharding"""
    print("\n=== SHARDING/PARTITIONING ===")

    db = ShardedDatabase(num_shards=4, strategy=ShardStrategy.HASH)

    # Write data across shards
    print("\nWriting user data across shards:")
    users = ["alice", "bob", "charlie", "diana", "eve", "frank", "grace", "henry"]
    for user in users:
        db.write(f"user:{user}", {"name": user, "email": f"{user}@example.com"})
        shard = db.get_shard(f"user:{user}")
        print(f"  user:{user} -> {shard.name}")

    # Show distribution
    stats = db.get_stats()
    print(f"\nSharding stats: {stats}")


def demonstrate_caching():
    """Demonstrate caching"""
    print("\n=== CACHING ===")

    cache = Cache(max_size=3, policy=CacheEvictionPolicy.LRU)

    # Simulate cache usage
    print("\nCache operations:")
    cache.put("user:1", {"name": "Alice"})
    cache.put("user:2", {"name": "Bob"})
    cache.put("user:3", {"name": "Charlie"})
    print("  Added 3 users to cache")

    # Cache hits
    print(f"  Get user:1: {cache.get('user:1')}")
    print(f"  Get user:2: {cache.get('user:2')}")

    # Cache miss
    print(f"  Get user:99: {cache.get('user:99')}")

    # Trigger eviction
    cache.put("user:4", {"name": "Diana"})
    print("  Added user:4 (should evict user:3 - LRU)")

    print(f"  Get user:3: {cache.get('user:3')} (evicted)")

    stats = cache.get_stats()
    print(f"\nCache stats: {stats}")


def demonstrate_async_processing():
    """Demonstrate async task processing"""
    print("\n=== ASYNC PROCESSING ===")

    queue = TaskQueue()

    # Enqueue tasks
    print("\nEnqueuing background tasks:")
    task_ids = [
        queue.enqueue("send_email", {"email": "user@example.com"}),
        queue.enqueue("generate_report", {"report_name": "monthly_sales"}),
        queue.enqueue("process_image", {"image_id": "img_123"}),
    ]
    print(f"  Enqueued {len(task_ids)} tasks")

    # Process tasks
    print("\nProcessing tasks:")
    processed = queue.process_batch(batch_size=5)
    print(f"  Processed {processed} tasks")

    # Check status
    for task_id in task_ids:
        status = queue.get_task_status(task_id)
        print(f"  {task_id}: {status.value if status else 'not found'}")

    stats = queue.get_stats()
    print(f"\nQueue stats: {stats}")


def demonstrate_read_replicas():
    """Demonstrate read replicas"""
    print("\n=== DATABASE READ REPLICAS ===")

    db = ReplicatedDatabase(num_replicas=3)

    # Write data
    print("\nWriting data to master:")
    db.write("product:1", {"name": "Laptop", "price": 999})
    db.write("product:2", {"name": "Mouse", "price": 29})
    print("  Wrote 2 products (replicated to all replicas)")

    # Read from replicas
    print("\nReading from replicas (load balanced):")
    for i in range(6):
        product = db.read("product:1")
        print(f"  Read {i+1}: {product}")

    stats = db.get_stats()
    print(f"\nReplication stats: {stats}")


def main():
    """Run all demonstrations"""
    print("SCALABILITY PATTERNS IN DISTRIBUTED SYSTEMS")
    print("=" * 60)

    demonstrate_load_balancing()
    demonstrate_sharding()
    demonstrate_caching()
    demonstrate_async_processing()
    demonstrate_read_replicas()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("1. Load balancing - Distribute work across multiple servers")
    print("2. Sharding - Partition data for horizontal scaling")
    print("3. Caching - Reduce load by storing frequently accessed data")
    print("4. Async processing - Handle work in background queues")
    print("5. Read replicas - Scale read operations independently")


if __name__ == "__main__":
    main()
