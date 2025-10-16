"""
Non-Functional Requirements in Software Systems

Non-functional requirements (NFRs) define HOW a system should behave, rather than WHAT it should do.
They specify quality attributes, constraints, and system properties that affect user experience and operations.

Categories of NFRs:
1. Performance - Speed, throughput, resource usage
2. Security - Authentication, authorization, data protection
3. Reliability - Uptime, fault tolerance, error handling
4. Scalability - Ability to handle growth
5. Maintainability - Code quality, documentation, testability
6. Usability - User experience, accessibility
7. Observability - Monitoring, logging, metrics
"""

import time
import random
import hashlib
import secrets
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from functools import wraps


# 1. PERFORMANCE REQUIREMENTS
@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    response_time_ms: float
    throughput_ops_per_sec: float
    cpu_usage_percent: float
    memory_usage_mb: float

    def meets_requirements(self, max_response_time_ms: float = 100, min_throughput: float = 1000) -> bool:
        """Check if metrics meet performance requirements"""
        return (
            self.response_time_ms <= max_response_time_ms and
            self.throughput_ops_per_sec >= min_throughput
        )


class PerformanceMonitor:
    """
    Performance Requirements: System must respond within acceptable time limits.
    Example NFRs:
    - API response time < 100 ms for 95th percentile
    - Support 10,000 requests per second
    - Database query time < 50 ms
    """

    def __init__(self, max_response_time_ms: float = 100):
        self.max_response_time_ms = max_response_time_ms
        self.measurements: list[float] = []

    def measure(self, func: Callable, *args, **kwargs) -> tuple[Any, float]:
        """Measure function execution time"""
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = (time.time() - start_time) * 1000

        self.measurements.append(elapsed_ms)
        return result, elapsed_ms

    def get_percentile(self, percentile: int = 95) -> float:
        """Get percentile response time"""
        if not self.measurements:
            return 0.0

        sorted_measurements = sorted(self.measurements)
        index = int(len(sorted_measurements) * percentile / 100)
        return sorted_measurements[min(index, len(sorted_measurements) - 1)]

    def verify_performance_requirements(self) -> dict:
        """Verify if performance requirements are met"""
        p50 = self.get_percentile(50)
        p95 = self.get_percentile(95)
        p99 = self.get_percentile(99)
        avg = sum(self.measurements) / len(self.measurements) if self.measurements else 0

        return {
            "average_ms": f"{avg:.2f}",
            "p50_ms": f"{p50:.2f}",
            "p95_ms": f"{p95:.2f}",
            "p99_ms": f"{p99:.2f}",
            "meets_p95_requirement": p95 <= self.max_response_time_ms,
            "requirement": f"< {self.max_response_time_ms}ms"
        }


# 2. SECURITY REQUIREMENTS
class AuthenticationLevel(Enum):
    """Authentication security levels"""
    NONE = "none"
    BASIC = "basic"
    MFA = "mfa"  # Multifactor authentication
    CERTIFICATE = "certificate"


@dataclass
class User:
    """User with authentication credentials"""
    user_id: str
    username: str
    password_hash: str
    role: str = "user"
    mfa_enabled: bool = False
    failed_login_attempts: int = 0
    last_login: Optional[datetime] = None


class SecurityManager:
    """
    Security Requirements: System must protect data and prevent unauthorized access.
    Example NFRs:
    - All passwords must be hashed with bcrypt/argon2
    - Support multifactor authentication
    - Lock accounts after 5 failed login attempts
    - All API endpoints require authentication
    - Encrypt data in transit (TLS) and at rest
    - Session timeout after 30 minutes of inactivity
    """

    def __init__(self, max_failed_attempts: int = 5, session_timeout_minutes: int = 30):
        self.max_failed_attempts = max_failed_attempts
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.users: dict[str, User] = {}
        self.sessions: dict[str, tuple[str, datetime]] = {}  # token -> (user_id, created_at)

    def register_user(self, username: str, password: str, role: str = "user") -> User:
        """Register a new user with secure password hashing"""
        # Hash password (simplified - production should use bcrypt/argon2)
        password_hash = hashlib.sha256(f"{password}:salt".encode()).hexdigest()

        user = User(
            user_id=f"user_{len(self.users)}",
            username=username,
            password_hash=password_hash,
            role=role
        )
        self.users[username] = user
        return user

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token"""
        if username not in self.users:
            return None

        user = self.users[username]

        # Check if the account is locked
        if user.failed_login_attempts >= self.max_failed_attempts:
            print(f"  [SECURITY] Account locked: {username}")
            return None

        # Verify password
        password_hash = hashlib.sha256(f"{password}:salt".encode()).hexdigest()
        if password_hash != user.password_hash:
            user.failed_login_attempts += 1
            print(f"  [SECURITY] Failed login attempt {user.failed_login_attempts}/{self.max_failed_attempts}")
            return None

        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.last_login = datetime.now()

        # Create a session token
        token = secrets.token_urlsafe(32)
        self.sessions[token] = (user.user_id, datetime.now())

        return token

    def validate_session(self, token: str) -> bool:
        """Validate session token and check timeout"""
        if token not in self.sessions:
            return False

        user_id, created_at = self.sessions[token]

        # Check session timeout
        if datetime.now() - created_at > self.session_timeout:
            del self.sessions[token]
            print(f"  [SECURITY] Session expired")
            return False

        return True

    def require_authentication(self, func: Callable) -> Callable:
        """Decorator to require authentication for functions"""
        @wraps(func)
        def wrapper(token: str, *args, **kwargs):
            if not self.validate_session(token):
                raise PermissionError("Authentication required")
            return func(*args, **kwargs)
        return wrapper

    def get_security_report(self) -> dict:
        """Get security metrics report"""
        locked_accounts = sum(1 for u in self.users.values() if u.failed_login_attempts >= self.max_failed_attempts)
        active_sessions = len(self.sessions)

        return {
            "total_users": len(self.users),
            "locked_accounts": locked_accounts,
            "active_sessions": active_sessions,
            "session_timeout_minutes": self.session_timeout.seconds // 60,
            "max_failed_attempts": self.max_failed_attempts
        }


# 3. RELIABILITY REQUIREMENTS
class ServiceStatus(Enum):
    """Service status"""
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"


@dataclass
class UptimeRecord:
    """Record of uptime/downtime"""
    timestamp: datetime
    status: ServiceStatus
    duration_seconds: float = 0.0


class ReliabilityMonitor:
    """
    Reliability Requirements: System must be available and handle failures gracefully.
    Example NFRs:
    - 99.9% uptime (8/76 hours downtime per year)
    - Automatic failover within 30 seconds
    - Zero data loss during failures
    - Graceful degradation under load
    - All errors logged and monitored
    """

    def __init__(self, target_uptime_percent: float = 99.9):
        self.target_uptime_percent = target_uptime_percent
        self.uptime_records: list[UptimeRecord] = []
        self.error_count = 0
        self.total_requests = 0
        self.start_time = datetime.now()

    def record_uptime(self, status: ServiceStatus, duration_seconds: float):
        """Record uptime/downtime event"""
        record = UptimeRecord(
            timestamp=datetime.now(),
            status=status,
            duration_seconds=duration_seconds
        )
        self.uptime_records.append(record)

    def record_request(self, success: bool):
        """Record request result"""
        self.total_requests += 1
        if not success:
            self.error_count += 1

    def calculate_uptime_percentage(self) -> float:
        """Calculate uptime percentage"""
        if not self.uptime_records:
            return 100.0

        total_time = sum(r.duration_seconds for r in self.uptime_records)
        uptime = sum(r.duration_seconds for r in self.uptime_records if r.status == ServiceStatus.UP)

        if total_time == 0:
            return 100.0

        return (uptime / total_time) * 100

    def calculate_error_rate(self) -> float:
        """Calculate error rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.error_count / self.total_requests) * 100

    def meets_reliability_requirements(self) -> bool:
        """Check if reliability requirements are met"""
        uptime = self.calculate_uptime_percentage()
        error_rate = self.calculate_error_rate()

        return uptime >= self.target_uptime_percent and error_rate < 1.0

    def get_reliability_report(self) -> dict:
        """Get the reliability metrics report"""
        uptime = self.calculate_uptime_percentage()
        error_rate = self.calculate_error_rate()

        # Calculate allowed downtime
        allowed_downtime_hours = (100 - self.target_uptime_percent) / 100 * 8760  # hours per year

        return {
            "uptime_percent": f"{uptime:.3f}%",
            "target_uptime": f"{self.target_uptime_percent}%",
            "error_rate": f"{error_rate:.2f}%",
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "allowed_downtime_per_year": f"{allowed_downtime_hours:.2f} hours",
            "meets_requirements": self.meets_reliability_requirements()
        }


# 4. SCALABILITY REQUIREMENTS
@dataclass
class LoadTestResult:
    """Result of load testing"""
    concurrent_users: int
    requests_per_second: float
    avg_response_time_ms: float
    error_rate_percent: float
    success: bool


class ScalabilityTester:
    """
    Scalability Requirements: System must handle growth in users, data, and load.
    Example NFRs:
    - Support 100,000 concurrent users
    - Scale horizontally by adding servers
    - Handle 10x traffic increase during peak times
    - Database can store 1TB+ of data
    - Response time degrades gracefully under load
    """

    def __init__(self, max_acceptable_response_time_ms: float = 500):
        self.max_acceptable_response_time_ms = max_acceptable_response_time_ms
        self.load_tests: list[LoadTestResult] = []

    def simulate_load(self, concurrent_users: int, duration_seconds: float = 1.0) -> LoadTestResult:
        """Simulate a load with specified concurrent users"""
        # Simulate request processing
        total_requests = int(concurrent_users * duration_seconds * 10)  # 10 req/sec per user
        errors = 0
        response_times = []

        for _ in range(min(total_requests, 100)):  # Sample 100 requests
            # Simulate response time degradation with a load
            base_time = 10  # 10 ms base
            load_factor = 1 + (concurrent_users / 1000)  # Degrades with a load
            response_time = base_time * load_factor + random.uniform(-5, 5)
            response_times.append(response_time)

            # Simulate errors under heavy load
            if concurrent_users > 5000 and random.random() < 0.05:
                errors += 1

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        error_rate = (errors / len(response_times) * 100) if response_times else 0
        rps = total_requests / duration_seconds

        result = LoadTestResult(
            concurrent_users=concurrent_users,
            requests_per_second=rps,
            avg_response_time_ms=avg_response_time,
            error_rate_percent=error_rate,
            success=avg_response_time <= self.max_acceptable_response_time_ms and error_rate < 1.0
        )
        self.load_tests.append(result)
        return result

    def find_breaking_point(self, start_users: int = 100, step: int = 500, max_users: int = 10000) -> int:
        """Find the breaking point where system performance degrades"""
        for users in range(start_users, max_users + 1, step):
            result = self.simulate_load(users)
            if not result.success:
                return users
        return max_users

    def get_scalability_report(self) -> dict:
        """Get the scalability test report"""
        if not self.load_tests:
            return {"tests_run": 0}

        max_successful_users = max(
            (t.concurrent_users for t in self.load_tests if t.success),
            default=0
        )

        return {
            "tests_run": len(self.load_tests),
            "max_successful_concurrent_users": max_successful_users,
            "max_acceptable_response_time_ms": self.max_acceptable_response_time_ms,
            "recommendation": f"System can handle up to {max_successful_users} concurrent users"
        }


# 5. MAINTAINABILITY REQUIREMENTS
class CodeQualityMetrics:
    """
    Maintainability Requirements: Code should be easy to understand, modify, and test.
    Example NFRs:
    - Code coverage > 80%
    - Cyclomatic complexity < 10 per function
    - All public APIs documented
    - Follow consistent coding standards
    - Technical debt tracked and addressed
    - Average time to fix bugs < 2 days
    """

    @staticmethod
    def calculate_cyclomatic_complexity(func: Callable) -> int:
        """Calculate cyclomatic complexity (simplified)"""
        import inspect
        source = inspect.getsource(func)

        # Count decision points (if, for, while, and, or, etc.)
        decision_keywords = ['if', 'elif', 'for', 'while', 'and', 'or', 'except']
        complexity = 1  # Base complexity

        for keyword in decision_keywords:
            complexity += source.count(f' {keyword} ')
            complexity += source.count(f'\n{keyword} ')

        return complexity

    @staticmethod
    def check_function_documentation(func: Callable) -> bool:
        """Check if the function has documentation"""
        return func.__doc__ is not None and len(func.__doc__.strip()) > 0

    @staticmethod
    def get_maintainability_score(
        code_coverage: float,
        avg_complexity: float,
        documentation_percent: float
    ) -> dict:
        """Calculate overall maintainability score"""
        # Scoring: 0-100
        coverage_score = min(code_coverage, 100)
        complexity_max = 100 - int(avg_complexity - 5) * 10
        complexity_score = max(0, complexity_max)  # Penalize complexity > 5
        doc_score = documentation_percent

        overall_score = (coverage_score + complexity_score + doc_score) / 3

        return {
            "code_coverage_percent": code_coverage,
            "avg_cyclomatic_complexity": avg_complexity,
            "documentation_percent": documentation_percent,
            "overall_maintainability_score": f"{overall_score:.1f}/100",
            "grade": "A" if overall_score >= 90 else "B" if overall_score >= 80 else "C" if overall_score >= 70 else "D"
        }


# 6. USABILITY REQUIREMENTS
@dataclass
class UsabilityMetrics:
    """Usability measurement results"""
    task_success_rate: float
    avg_task_completion_time_seconds: float
    user_satisfaction_score: float  # 1-5 scale
    error_recovery_rate: float


class UsabilityEvaluator:
    """
    Usability Requirements: System should be easy and intuitive to use.
    Example NFRs:
    - New users can complete core tasks within 5 minutes
    - Task success rate > 90%
    - User satisfaction score > 4.0/5.0
    - Accessible to users with disabilities (WCAG 2.1 AA)
    - Support internationalization (i18n)
    - Mobile-responsive design
    """

    def __init__(self):
        self.task_results: list[tuple[bool, float]] = []  # (success, time)
        self.satisfaction_scores: list[float] = []

    def record_task(self, success: bool, completion_time_seconds: float):
        """Record user task result"""
        self.task_results.append((success, completion_time_seconds))

    def record_satisfaction(self, score: float):
        """Record user satisfaction score (1-5)"""
        if 1 <= score <= 5:
            self.satisfaction_scores.append(score)

    def evaluate_usability(self) -> UsabilityMetrics:
        """Evaluate overall usability"""
        if not self.task_results:
            return UsabilityMetrics(0, 0, 0, 0)

        successful_tasks = [t for t in self.task_results if t[0]]
        success_rate = len(successful_tasks) / len(self.task_results) * 100

        avg_time = sum(t[1] for t in self.task_results) / len(self.task_results)

        avg_satisfaction = sum(self.satisfaction_scores) / len(self.satisfaction_scores) if self.satisfaction_scores else 0

        # Simulate error recovery (users who recovered from errors)
        error_recovery_rate = 85.0  # Placeholder

        return UsabilityMetrics(
            task_success_rate=success_rate,
            avg_task_completion_time_seconds=avg_time,
            user_satisfaction_score=avg_satisfaction,
            error_recovery_rate=error_recovery_rate
        )

    def meets_usability_requirements(self) -> bool:
        """Check if usability requirements are met"""
        metrics = self.evaluate_usability()
        return (
            metrics.task_success_rate >= 90 and
            metrics.user_satisfaction_score >= 4.0 and
            metrics.avg_task_completion_time_seconds <= 300  # 5 minutes
        )


# 7. OBSERVABILITY REQUIREMENTS
class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Log entry"""
    timestamp: datetime
    level: LogLevel
    message: str
    context: dict = field(default_factory=dict)


class ObservabilitySystem:
    """
    Observability Requirements: System must be monitorable and debuggable.
    Example NFRs:
    - All errors logged with context
    - Distributed tracing for requests
    - Metrics exported in Prometheus format
    - Alerts for critical issues within 1 minute
    - Log retention for 90 days
    - Dashboard for system health visualization
    """

    def __init__(self):
        self.logs: list[LogEntry] = []
        self.metrics: dict[str, list[float]] = {}
        self.alerts_triggered = 0

    def log(self, level: LogLevel, message: str, context: dict = None):
        """Log a message with context"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            context=context or {}
        )
        self.logs.append(entry)

        # Trigger alert for critical issues
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self._trigger_alert(message)

    def _trigger_alert(self, message: str):
        """Trigger an alert for critical issues"""
        self.alerts_triggered += 1
        print(f"  [ALERT] {message}")

    def record_metric(self, metric_name: str, value: float):
        """Record a metric value"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def get_metric_stats(self, metric_name: str) -> dict:
        """Get statistics for a metric"""
        if metric_name not in self.metrics:
            return {}

        values = self.metrics[metric_name]
        return {
            "count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values)
        }

    def get_observability_report(self) -> dict:
        """Get observability report"""
        log_counts = {}
        for level in LogLevel:
            count = sum(1 for log in self.logs if log.level == level)
            log_counts[level.value] = count

        return {
            "total_logs": len(self.logs),
            "logs_by_level": log_counts,
            "metrics_tracked": len(self.metrics),
            "alerts_triggered": self.alerts_triggered
        }


def demonstrate_performance_requirements():
    """Demonstrate performance requirements"""
    print("\n=== PERFORMANCE REQUIREMENTS ===")
    print("NFR: API response time < 100ms for 95th percentile\n")

    monitor = PerformanceMonitor(max_response_time_ms=100)

    def api_endpoint():
        """Simulate API endpoint"""
        time.sleep(random.uniform(0.01, 0.15))  # 10-150ms
        return {"status": "success"}

    print("Running 100 API requests:")
    for _ in range(100):
        monitor.measure(api_endpoint)

    report = monitor.verify_performance_requirements()
    print(f"Performance Report: {report}")

    if report["meets_p95_requirement"]:
        print("✓ Performance requirements MET")
    else:
        print("✗ Performance requirements NOT MET")


def demonstrate_security_requirements():
    """Demonstrate security requirements"""
    print("\n=== SECURITY REQUIREMENTS ===")
    print("NFRs: Password hashing, account lockout, session management\n")

    security = SecurityManager(max_failed_attempts=3, session_timeout_minutes=30)

    # Register users
    print("Registering users:")
    security.register_user("alice", "SecurePass123!", role="admin")
    security.register_user("bob", "AnotherPass456!", role="user")

    # Successful login
    print("\nAlice logging in:")
    token = security.authenticate("alice", "SecurePass123!")
    if token:
        print(f"  ✓ Login successful, token: {token[:20]}...")

    # Failed login attempts
    print("\nBob attempting wrong passwords:")
    for i in range(4):
        result = security.authenticate("bob", "WrongPassword")
        if result is None:
            print(f"  Attempt {i+1} failed")

    print(f"\nSecurity Report: {security.get_security_report()}")


def demonstrate_reliability_requirements():
    """Demonstrate reliability requirements"""
    print("\n=== RELIABILITY REQUIREMENTS ===")
    print("NFR: 99.9% uptime (8.76 hours downtime per year)\n")

    monitor = ReliabilityMonitor(target_uptime_percent=99.9)

    # Simulate uptime/downtime
    print("Simulating service operation:")
    monitor.record_uptime(ServiceStatus.UP, duration_seconds=86400)  # 1 day up
    monitor.record_uptime(ServiceStatus.DOWN, duration_seconds=60)  # 1 minute down
    monitor.record_uptime(ServiceStatus.UP, duration_seconds=86400)  # 1 day up

    # Record requests
    for _ in range(1000):
        success = random.random() > 0.005  # 0.5% error rate
        monitor.record_request(success)

    report = monitor.get_reliability_report()
    print(f"Reliability Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")


def demonstrate_scalability_requirements():
    """Demonstrate scalability requirements"""
    print("\n=== SCALABILITY REQUIREMENTS ===")
    print("NFR: Support 10,000 concurrent users with < 500ms response time\n")

    tester = ScalabilityTester(max_acceptable_response_time_ms=500)

    print("Load testing with increasing concurrent users:")
    for users in [100, 1000, 5000, 10000]:
        result = tester.simulate_load(users, duration_seconds=1.0)
        status = "✓ PASS" if result.success else "✗ FAIL"
        print(f"  {users:,} users: {result.avg_response_time_ms:.1f}ms avg, "
              f"{result.error_rate_percent:.2f}% errors [{status}]")

    print(f"\nScalability Report: {tester.get_scalability_report()}")


def demonstrate_maintainability_requirements():
    """Demonstrate maintainability requirements"""
    print("\n=== MAINTAINABILITY REQUIREMENTS ===")
    print("NFRs: Code coverage > 80%, complexity < 10, all APIs documented\n")

    # Example function to analyze
    def example_function(x: int, y: int) -> int:
        """Add two numbers together."""
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        return x + y

    complexity = CodeQualityMetrics.calculate_cyclomatic_complexity(example_function)
    has_docs = CodeQualityMetrics.check_function_documentation(example_function)

    print(f"Function Analysis:")
    print(f"  Cyclomatic Complexity: {complexity}")
    print(f"  Has Documentation: {has_docs}")

    # Overall maintainability score
    score = CodeQualityMetrics.get_maintainability_score(
        code_coverage=85.0,
        avg_complexity=6.5,
        documentation_percent=90.0
    )

    print(f"\nMaintainability Score:")
    for key, value in score.items():
        print(f"  {key}: {value}")


def demonstrate_usability_requirements():
    """Demonstrate usability requirements"""
    print("\n=== USABILITY REQUIREMENTS ===")
    print("NFRs: 90% task success rate, < 5 min completion time, 4.0+ satisfaction\n")

    evaluator = UsabilityEvaluator()

    # Simulate user tasks
    print("Simulating user tasks:")
    tasks = [
        (True, 120),   # Success, 2 minutes
        (True, 180),   # Success, 3 minutes
        (False, 300),  # Failed, 5 minutes
        (True, 90),    # Success, 1.5 minutes
        (True, 150),   # Success, 2.5 minutes
    ]

    for success, time_sec in tasks:
        evaluator.record_task(success, time_sec)

    # User satisfaction scores
    for score in [4.5, 4.0, 5.0, 4.2, 4.8]:
        evaluator.record_satisfaction(score)

    metrics = evaluator.evaluate_usability()
    print(f"\nUsability Metrics:")
    print(f"  Task Success Rate: {metrics.task_success_rate:.1f}%")
    print(f"  Avg Completion Time: {metrics.avg_task_completion_time_seconds:.1f}s")
    print(f"  User Satisfaction: {metrics.user_satisfaction_score:.1f}/5.0")
    print(f"  Error Recovery Rate: {metrics.error_recovery_rate:.1f}%")

    if evaluator.meets_usability_requirements():
        print("\n✓ Usability requirements MET")
    else:
        print("\n✗ Usability requirements NOT MET")


def demonstrate_observability_requirements():
    """Demonstrate observability requirements"""
    print("\n=== OBSERVABILITY REQUIREMENTS ===")
    print("NFRs: All errors logged, metrics tracked, alerts for critical issues\n")

    obs = ObservabilitySystem()

    # Log various events
    print("Logging system events:")
    obs.log(LogLevel.INFO, "Application started", {"version": "1.0.0"})
    obs.log(LogLevel.INFO, "User logged in", {"user_id": "user_123"})
    obs.log(LogLevel.WARNING, "High memory usage detected", {"memory_mb": 1500})
    obs.log(LogLevel.ERROR, "Database connection failed", {"error": "timeout"})
    obs.log(LogLevel.CRITICAL, "Service crashed", {"exit_code": 1})

    # Record metrics
    print("\nRecording metrics:")
    for _ in range(10):
        obs.record_metric("response_time_ms", random.uniform(50, 200))
        obs.record_metric("requests_per_second", random.uniform(100, 500))

    # Get metrics stats
    response_time_stats = obs.get_metric_stats("response_time_ms")
    print(f"  Response Time Stats: {response_time_stats}")

    # Observability report
    report = obs.get_observability_report()
    print(f"\nObservability Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")


def main():
    """Run all demonstrations"""
    print("NON-FUNCTIONAL REQUIREMENTS IN SOFTWARE SYSTEMS")
    print("=" * 70)
    print("\nNon-functional requirements define QUALITY ATTRIBUTES and CONSTRAINTS")
    print("that affect how the system operates, rather than what it does.\n")

    demonstrate_performance_requirements()
    demonstrate_security_requirements()
    demonstrate_reliability_requirements()
    demonstrate_scalability_requirements()
    demonstrate_maintainability_requirements()
    demonstrate_usability_requirements()
    demonstrate_observability_requirements()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("1. Performance - Define response times, throughput, resource usage")
    print("2. Security - Authentication, authorization, encryption")
    print("3. Reliability - Uptime targets, fault tolerance, error handling")
    print("4. Scalability - User growth, data growth, load handling")
    print("5. Maintainability - Code quality, testing, documentation")
    print("6. Usability - User experience, accessibility, ease of use")
    print("7. Observability - Logging, metrics, monitoring, alerting")
    print("\nNFRs should be SPECIFIC, MEASURABLE, and TESTABLE")


if __name__ == "__main__":
    main()
