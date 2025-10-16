"""
Performance Patterns in Distributed Systems

Performance refers to how efficiently a system processes requests and utilizes resources.
Key metrics: latency (response time), throughput (requests/second), resource utilization.

Key performance patterns:
1. Connection Pooling - Reuse connections to avoid overhead
2. Lazy Loading - Defer expensive operations until needed
3. Request Batching - Group multiple operations into one
4. Data Compression - Reduce network transfer size
5. Parallel Processing - Execute tasks concurrently
"""

import time
import random
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from threading import Lock
import gzip
import json


# 1. CONNECTION POOLING
@dataclass
class Connection:
    """A connection to a resource (database, API, etc.)"""
    conn_id: str
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    in_use: bool = False
    use_count: int = 0

    def __hash__(self):
        return hash(self.conn_id)

    def __eq__(self, other):
        if not isinstance(other, Connection):
            return NotImplemented
        return self.conn_id == other.conn_id

    def execute(self, query: str) -> str:
        """Execute a query using this connection"""
        if not self.in_use:
            raise RuntimeError("Connection not acquired")

        self.use_count += 1
        self.last_used = time.time()

        # Simulate query execution
        time.sleep(0.001)
        return f"Result for: {query}"

    def get_age(self) -> float:
        """Get connection age in seconds"""
        return time.time() - self.created_at


class ConnectionPool:
    """
    Connection pool maintains reusable connections to avoid expensive
    connection establishment overhead.
    """

    def __init__(self, min_size: int = 5, max_size: int = 20, max_idle_time: float = 300):
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time

        self.available: list[Connection] = []
        self.in_use: set[Connection] = set()
        self.lock = Lock()
        self.conn_counter = 0
        self.total_connections_created = 0

        # Initialize minimum connections
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the pool with minimum connections"""
        for _ in range(self.min_size):
            conn = self._create_connection()
            self.available.append(conn)

    def _create_connection(self) -> Connection:
        """Create a new connection"""
        self.conn_counter += 1
        self.total_connections_created += 1
        return Connection(conn_id=f"conn_{self.conn_counter}")

    def acquire(self, timeout: float = 5.0) -> Optional[Connection]:
        """Acquire a connection from the pool"""
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            with self.lock:
                # Reuse available connection
                if self.available:
                    conn = self.available.pop(0)
                    conn.in_use = True
                    self.in_use.add(conn)
                    return conn

                # Create a new connection if under max size
                if len(self.in_use) < self.max_size:
                    conn = self._create_connection()
                    conn.in_use = True
                    self.in_use.add(conn)
                    return conn

            # Wait briefly before retrying
            time.sleep(0.01)

        return None

    def release(self, conn: Connection):
        """Release a connection back to the pool"""
        with self.lock:
            if conn in self.in_use:
                conn.in_use = False
                self.in_use.remove(conn)

                # Check if the connection is still valid
                if conn.get_age() < self.max_idle_time:
                    self.available.append(conn)

    def cleanup_idle_connections(self):
        """Remove connections that have been idle too long"""
        with self.lock:
            self.available = [
                conn for conn in self.available
                if (time.time() - conn.last_used) < self.max_idle_time
            ]

            # Maintain the minimum pool size
            while len(self.available) < self.min_size:
                self.available.append(self._create_connection())

    def get_stats(self) -> dict:
        """Get pool statistics"""
        with self.lock:
            return {
                "available": len(self.available),
                "in_use": len(self.in_use),
                "total": len(self.available) + len(self.in_use),
                "total_created": self.total_connections_created
            }


# 2. LAZY LOADING
class LazyResource:
    """
    Lazy loading defers expensive initialization until first use.
    Improves startup time and avoids unnecessary work.
    """

    def __init__(self, name: str, load_fn: Callable):
        self.name = name
        self.load_fn = load_fn
        self._data: Optional[Any] = None
        self._loaded = False
        self.load_time: Optional[float] = None

    def is_loaded(self) -> bool:
        """Check if a resource is loaded"""
        return self._loaded

    def get(self) -> Any:
        """Get resource, loading if necessary"""
        if not self._loaded:
            print(f"  [LAZY LOAD] Loading {self.name}...")
            start_time = time.time()
            self._data = self.load_fn()
            self.load_time = time.time() - start_time
            self._loaded = True
            print(f"  [LAZY LOAD] {self.name} loaded in {self.load_time*1000:.1f}ms")

        return self._data

    def invalidate(self):
        """Invalidate cached resource"""
        self._data = None
        self._loaded = False


class LazyApplication:
    """
    Application with lazy-loaded resources.
    Only loads resources when they're actually needed.
    """

    def __init__(self):
        self.resources = {
            "database": LazyResource("database", self._load_database),
            "cache": LazyResource("cache", self._load_cache),
            "search_index": LazyResource("search_index", self._load_search_index),
            "ml_model": LazyResource("ml_model", self._load_ml_model)
        }

    def _load_database(self) -> dict:
        """Simulate expensive database initialization"""
        time.sleep(0.05)
        return {"connection": "db://localhost:5432"}

    def _load_cache(self) -> dict:
        """Simulate cache initialization"""
        time.sleep(0.02)
        return {"connection": "redis://localhost:6379"}

    def _load_search_index(self) -> dict:
        """Simulate search index loading"""
        time.sleep(0.1)
        return {"index": "elasticsearch", "docs": 1000000}

    def _load_ml_model(self) -> dict:
        """Simulate ML model loading"""
        time.sleep(0.15)
        return {"model": "bert-base", "size": "440MB"}

    def get_resource(self, name: str) -> Any:
        """Get a resource (lazy loaded)"""
        if name in self.resources:
            return self.resources[name].get()
        return None

    def get_loaded_resources(self) -> list[str]:
        """Get the list of loaded resources"""
        return [name for name, res in self.resources.items() if res.is_loaded()]


# 3. REQUEST BATCHING
@dataclass
class BatchRequest:
    """A request to be batched"""
    request_id: str
    data: Any
    timestamp: float = field(default_factory=time.time)


class RequestBatcher:
    """
    Request batching groups multiple operations into one.
    Reduces overhead by processing requests in batches.
    """

    def __init__(self, batch_size: int = 10, max_wait_time: float = 0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time

        self.pending: list[BatchRequest] = []
        self.processed_batches = 0
        self.total_requests = 0

    def add_request(self, request_id: str, data: Any):
        """Add a request to the batch"""
        request = BatchRequest(request_id=request_id, data=data)
        self.pending.append(request)
        self.total_requests += 1

    def should_process_batch(self) -> bool:
        """Check if a batch should be processed"""
        if not self.pending:
            return False

        # Process if a batch is full
        if len(self.pending) >= self.batch_size:
            return True

        # Process if the oldest request has waited too long
        oldest = self.pending[0]
        if (time.time() - oldest.timestamp) >= self.max_wait_time:
            return True

        return False

    def process_batch(self) -> dict:
        """Process the current batch"""
        if not self.pending:
            return {"processed": 0, "requests": []}

        # Take batch
        batch = self.pending[:self.batch_size]
        self.pending = self.pending[self.batch_size:]

        # Process batch (simulated)
        start_time = time.time()
        results = []

        # Single expensive operation for the entire batch
        time.sleep(0.01)  # Batch overhead

        for req in batch:
            results.append({
                "request_id": req.request_id,
                "result": f"Processed: {req.data}",
                "wait_time_ms": (start_time - req.timestamp) * 1000
            })

        self.processed_batches += 1
        processing_time = (time.time() - start_time) * 1000

        return {
            "processed": len(batch),
            "requests": results,
            "processing_time_ms": processing_time
        }

    def get_stats(self) -> dict:
        """Get batching statistics"""
        avg_batch_size = self.total_requests / self.processed_batches if self.processed_batches > 0 else 0
        return {
            "pending": len(self.pending),
            "processed_batches": self.processed_batches,
            "total_requests": self.total_requests,
            "avg_batch_size": f"{avg_batch_size:.1f}"
        }


# 4. DATA COMPRESSION
class CompressionStats:
    """Statistics for compression operations"""

    def __init__(self):
        self.total_original_size = 0
        self.total_compressed_size = 0
        self.compression_time = 0.0
        self.decompression_time = 0.0
        self.operations = 0

    def add_operation(self, original_size: int, compressed_size: int,
                      compression_time: float, decompression_time: float):
        """Record a compression operation"""
        self.total_original_size += original_size
        self.total_compressed_size += compressed_size
        self.compression_time += compression_time
        self.decompression_time += decompression_time
        self.operations += 1

    def get_compression_ratio(self) -> float:
        """Calculate compression ratio"""
        if self.total_original_size == 0:
            return 0.0
        return self.total_compressed_size / self.total_original_size

    def get_space_saved(self) -> int:
        """Calculate bytes saved"""
        return self.total_original_size - self.total_compressed_size


class DataCompressor:
    """
    Data compression reduces network transfer size.
    Trades CPU time for reduced bandwidth usage.
    """

    def __init__(self):
        self.stats = CompressionStats()

    def compress(self, data: dict) -> bytes:
        """Compress data using gzip"""
        start_time = time.time()

        # Convert to JSON and compress
        json_data = json.dumps(data).encode('utf-8')
        compressed = gzip.compress(json_data)

        compression_time = time.time() - start_time

        # Update stats (decompression time recorded on decompress)
        self.stats.add_operation(
            len(json_data),
            len(compressed),
            compression_time,
            0.0
        )

        return compressed

    def decompress(self, compressed_data: bytes) -> dict:
        """Decompress data"""
        start_time = time.time()

        # Decompress and parse JSON
        json_data = gzip.decompress(compressed_data)
        data = json.loads(json_data.decode('utf-8'))

        decompression_time = time.time() - start_time

        # Update decompression time
        if self.stats.operations > 0:
            self.stats.decompression_time += decompression_time

        return data

    def get_stats(self) -> dict:
        """Get compression statistics"""
        ratio = self.stats.get_compression_ratio()
        saved = self.stats.get_space_saved()

        return {
            "operations": self.stats.operations,
            "original_size_bytes": self.stats.total_original_size,
            "compressed_size_bytes": self.stats.total_compressed_size,
            "compression_ratio": f"{ratio:.2%}",
            "space_saved_bytes": saved,
            "space_saved_percent": f"{(1-ratio)*100:.1f}%"
        }


# 5. PARALLEL PROCESSING
@dataclass
class Task:
    """A task to be processed"""
    task_id: str
    data: Any
    result: Optional[Any] = None
    processing_time: Optional[float] = None


class ParallelProcessor:
    """
    Parallel processing executes tasks concurrently.
    Improves throughput by utilizing multiple resources.
    """

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.tasks_processed = 0

    def process_task(self, task: Task) -> Task:
        """Process a single task"""
        start_time = time.time()

        # Simulate CPU-intensive work
        time.sleep(random.uniform(0.01, 0.03))

        task.result = f"Processed: {task.data}"
        task.processing_time = time.time() - start_time

        return task

    def process_sequential(self, tasks: list[Task]) -> list[Task]:
        """Process tasks sequentially (baseline)"""
        results = []
        for task in tasks:
            results.append(self.process_task(task))
        return results

    def process_parallel(self, tasks: list[Task]) -> list[Task]:
        """
        Simulate parallel processing.
        Note: Using time.sleep to simulate; a real implementation would use
        threading.Thread or multiprocessing.Process
        """
        # Simulate parallel execution by dividing work
        batch_size = max(1, len(tasks) // self.num_workers)
        results = []

        # Process in batches (simulating parallel workers)
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            # In real implementation, these would run in parallel
            for task in batch:
                results.append(self.process_task(task))

        return results

    @staticmethod
    def compare_performance(sequential_time: float, parallel_time: float, num_workers: int) -> dict:
        """Compare sequential vs parallel performance"""
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        efficiency = speedup / num_workers if num_workers > 0 else 0

        return {
            "sequential_time_ms": sequential_time * 1000,
            "parallel_time_ms": parallel_time * 1000,
            "speedup": f"{speedup:.2f}x",
            "efficiency": f"{efficiency*100:.1f}%"
        }


def demonstrate_connection_pooling():
    """Demonstrate connection pooling"""
    print("\n=== CONNECTION POOLING ===")

    pool = ConnectionPool(min_size=3, max_size=10)

    print("\nInitial pool state:")
    print(f"  Stats: {pool.get_stats()}")

    # Acquire and use connections
    print("\nAcquiring 5 connections:")
    connections = []
    for i in range(5):
        conn = pool.acquire()
        if conn:
            connections.append(conn)
            result = conn.execute(f"SELECT * FROM users WHERE id={i}")
            print(f"  {conn.conn_id}: {result}")

    print(f"\nPool stats after acquisition: {pool.get_stats()}")

    # Release connections back to the pool
    print("\nReleasing connections back to pool:")
    for conn in connections:
        pool.release(conn)

    print(f"Pool stats after release: {pool.get_stats()}")
    print(f"✓ Reused connections {pool.total_connections_created} times, created only {pool.total_connections_created} total")


def demonstrate_lazy_loading():
    """Demonstrate lazy loading"""
    print("\n=== LAZY LOADING ===")

    print("\nInitializing application (lazy loading enabled):")
    start_time = time.time()
    app = LazyApplication()
    init_time = (time.time() - start_time) * 1000
    print(f"  Application initialized in {init_time:.1f}ms")
    print(f"  Loaded resources: {app.get_loaded_resources()}")

    # Access resources as needed
    print("\nAccessing database (first time):")
    db = app.get_resource("database")
    print(f"  Database: {db}")

    print("\nAccessing cache (first time):")
    cache = app.get_resource("cache")
    print(f"  Cache: {cache}")

    print(f"\nLoaded resources: {app.get_loaded_resources()}")
    print("✓ Only loaded resources that were actually used")


def demonstrate_request_batching():
    """Demonstrate request batching"""
    print("\n=== REQUEST BATCHING ===")

    batcher = RequestBatcher(batch_size=5, max_wait_time=0.1)

    # Add requests
    print("\nAdding 12 requests:")
    for i in range(12):
        batcher.add_request(f"req_{i}", {"user_id": i, "action": "fetch_profile"})
        time.sleep(0.005)  # Small delay between requests

    print(f"  Added {batcher.total_requests} requests")

    # Process batches
    print("\nProcessing batches:")
    batch_num = 1
    while batcher.should_process_batch() or batcher.pending:
        if batcher.should_process_batch():
            result = batcher.process_batch()
            print(f"  Batch {batch_num}: processed {result['processed']} requests in {result['processing_time_ms']:.1f}ms")
            batch_num += 1

    stats = batcher.get_stats()
    print(f"\nBatching stats: {stats}")
    print(f"✓ Processed {stats['total_requests']} requests in {stats['processed_batches']} batches")


def demonstrate_data_compression():
    """Demonstrate data compression"""
    print("\n=== DATA COMPRESSION ===")

    compressor = DataCompressor()

    # Create sample data
    data = {
        "users": [
            {"id": i, "name": f"User{i}", "email": f"user{i}@example.com", "status": "active"}
            for i in range(100)
        ],
        "metadata": {
            "timestamp": "2024-01-15T10:30:00Z",
            "version": "1.0.0",
            "source": "api.example.com"
        }
    }

    print("\nCompressing data:")
    original_size = len(json.dumps(data).encode('utf-8'))
    print(f"  Original size: {original_size} bytes")

    compressed = compressor.compress(data)
    print(f"  Compressed size: {len(compressed)} bytes")

    # Decompress
    print("\nDecompressing data:")
    decompressed = compressor.decompress(compressed)
    print(f"  Decompressed successfully: {len(decompressed['users'])} users")

    stats = compressor.get_stats()
    print(f"\nCompression stats: {stats}")
    print(f"✓ Saved {stats['space_saved_percent']} of bandwidth")


def demonstrate_parallel_processing():
    """Demonstrate parallel processing"""
    print("\n=== PARALLEL PROCESSING ===")

    processor = ParallelProcessor(num_workers=4)

    # Create tasks
    tasks = [Task(task_id=f"task_{i}", data=f"data_{i}") for i in range(20)]

    # Sequential processing
    print("\nProcessing 20 tasks sequentially:")
    start_time = time.time()
    processor.process_sequential(tasks.copy())
    sequential_time = time.time() - start_time
    print(f"  Completed in {sequential_time*1000:.1f}ms")

    # Parallel processing (simulated)
    print("\nProcessing 20 tasks in parallel (4 workers):")
    start_time = time.time()
    processor.process_parallel(tasks.copy())
    parallel_time = time.time() - start_time
    print(f"  Completed in {parallel_time*1000:.1f}ms")

    # Compare
    comparison = ParallelProcessor.compare_performance(sequential_time, parallel_time, 4)
    print(f"\nPerformance comparison: {comparison}")
    print(f"✓ Parallel processing achieved {comparison['speedup']} speedup")


def main():
    """Run all demonstrations"""
    print("PERFORMANCE PATTERNS IN DISTRIBUTED SYSTEMS")
    print("=" * 60)

    demonstrate_connection_pooling()
    demonstrate_lazy_loading()
    demonstrate_request_batching()
    demonstrate_data_compression()
    demonstrate_parallel_processing()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("1. Connection pooling - Reuse expensive resources")
    print("2. Lazy loading - Defer work until needed")
    print("3. Request batching - Reduce per-request overhead")
    print("4. Data compression - Trade CPU for bandwidth")
    print("5. Parallel processing - Utilize multiple resources concurrently")
    print("\nOptimize the right things: Measure first, then optimize bottlenecks")


if __name__ == "__main__":
    main()