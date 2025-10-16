"""
Extensibility Patterns in Distributed Systems

Extensibility refers to a system's ability to accommodate new functionality
without modifying existing code. Key to long-term maintainability and evolution.

Key extensibility patterns:
1. Plugin Architecture - Add features through external modules
2. Service Registry - Dynamic service discovery and registration
3. Event-Driven Architecture - Loose coupling through events
4. API Versioning - Support multiple versions simultaneously
5. Feature Flags - Enable/disable features at runtime
"""

import time
from typing import Any, Callable, Optional, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum


# 1. PLUGIN ARCHITECTURE
class Plugin(ABC):
    """Base plugin interface"""

    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name"""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version"""
        pass

    @abstractmethod
    def initialize(self):
        """Initialize the plugin"""
        pass

    @abstractmethod
    def execute(self, context: dict) -> Any:
        """Execute plugin functionality"""
        pass


class PluginManager:
    """
    Plugin architecture allows extending functionality through external modules.
    New features can be added without modifying the core system.
    """

    def __init__(self):
        self.plugins: dict[str, Plugin] = {}
        self.execution_order: list[str] = []

    def register_plugin(self, plugin: Plugin):
        """Register a new plugin"""
        name = plugin.get_name()
        if name in self.plugins:
            raise ValueError(f"Plugin '{name}' already registered")

        print(f"  [PLUGIN] Registering {name} v{plugin.get_version()}")
        plugin.initialize()
        self.plugins[name] = plugin
        self.execution_order.append(name)

    def unregister_plugin(self, plugin_name: str):
        """Unregister a plugin"""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            self.execution_order.remove(plugin_name)
            print(f"  [PLUGIN] Unregistered {plugin_name}")

    def execute_plugin(self, plugin_name: str, context: dict) -> Any:
        """Execute a specific plugin"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found")

        return self.plugins[plugin_name].execute(context)

    def execute_all(self, context: dict) -> dict[str, Any]:
        """Execute all plugins in order"""
        results = {}
        for plugin_name in self.execution_order:
            plugin = self.plugins[plugin_name]
            results[plugin_name] = plugin.execute(context)
        return results

    def list_plugins(self) -> list[dict]:
        """List all registered plugins"""
        return [
            {
                "name": plugin.get_name(),
                "version": plugin.get_version()
            }
            for plugin in self.plugins.values()
        ]


# Example plugins
class AuthenticationPlugin(Plugin):
    """Plugin for authentication"""

    def get_name(self) -> str:
        return "authentication"

    def get_version(self) -> str:
        return "1.0.0"

    def initialize(self):
        print("    Authentication plugin initialized")

    def execute(self, context: dict) -> Any:
        user = context.get("user")
        return {"authenticated": user is not None, "user": user}


class LoggingPlugin(Plugin):
    """Plugin for logging"""

    def get_name(self) -> str:
        return "logging"

    def get_version(self) -> str:
        return "2.1.0"

    def initialize(self):
        print("    Logging plugin initialized")

    def execute(self, context: dict) -> Any:
        action = context.get("action", "unknown")
        print(f"    [LOG] Action: {action}")
        return {"logged": True, "action": action}


class MetricsPlugin(Plugin):
    """Plugin for metrics collection"""

    def get_name(self) -> str:
        return "metrics"

    def get_version(self) -> str:
        return "1.5.2"

    def initialize(self):
        self.request_count = 0

    def execute(self, context: dict) -> Any:
        self.request_count += 1
        return {"request_count": self.request_count}


# 2. SERVICE REGISTRY
@dataclass
class ServiceMetadata:
    """Metadata for a registered service"""
    service_id: str
    name: str
    version: str
    host: str
    port: int
    endpoints: list[str]
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def is_healthy(self, timeout: float = 30.0) -> bool:
        """Check if service is healthy based on heartbeat"""
        return (time.time() - self.last_heartbeat) < timeout


class ServiceRegistry:
    """
    Service registry enables dynamic service discovery.
    Services can register/deregister at runtime without central configuration.
    """

    def __init__(self):
        self.services: dict[str, ServiceMetadata] = {}
        self.service_types: dict[str, list[str]] = {}  # name -> [service_ids]

    def register(self, service: ServiceMetadata) -> str:
        """Register a new service"""
        service_id = service.service_id

        if service_id in self.services:
            print(f"  [REGISTRY] Updating service: {service_id}")
        else:
            print(f"  [REGISTRY] Registering service: {service_id} ({service.name} v{service.version})")

        self.services[service_id] = service

        # Track by service name
        if service.name not in self.service_types:
            self.service_types[service.name] = []
        if service_id not in self.service_types[service.name]:
            self.service_types[service.name].append(service_id)

        return service_id

    def deregister(self, service_id: str):
        """Deregister a service"""
        if service_id in self.services:
            service = self.services[service_id]
            del self.services[service_id]

            # Remove from service types
            if service.name in self.service_types:
                self.service_types[service.name].remove(service_id)
                if not self.service_types[service.name]:
                    del self.service_types[service.name]

            print(f"  [REGISTRY] Deregistered service: {service_id}")

    def heartbeat(self, service_id: str):
        """Update service heartbeat"""
        if service_id in self.services:
            self.services[service_id].last_heartbeat = time.time()

    def discover(self, service_name: str) -> list[ServiceMetadata]:
        """Discover all healthy instances of a service"""
        if service_name not in self.service_types:
            return []

        service_ids = self.service_types[service_name]
        return [
            self.services[sid]
            for sid in service_ids
            if sid in self.services and self.services[sid].is_healthy()
        ]

    def get_service(self, service_id: str) -> Optional[ServiceMetadata]:
        """Get a specific service by ID"""
        return self.services.get(service_id)

    def list_services(self) -> list[ServiceMetadata]:
        """List all registered services"""
        return list(self.services.values())


# 3. EVENT-DRIVEN ARCHITECTURE
@dataclass
class Event:
    """An event in the system"""
    event_type: str
    data: dict
    timestamp: float = field(default_factory=time.time)
    source: str = "system"


class EventHandler(Protocol):
    """Protocol for event handlers"""

    def __call__(self, event: Event) -> None:
        """Handle an event"""
        ...


class EventBus:
    """
    Event-driven architecture enables loose coupling.
    Components communicate through events without direct dependencies.
    """

    def __init__(self):
        self.subscribers: dict[str, list[EventHandler]] = {}
        self.event_history: list[Event] = []

    def subscribe(self, event_type: str, handler: EventHandler):
        """Subscribe to an event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []

        self.subscribers[event_type].append(handler)
        handler_name = getattr(handler, '__name__', str(handler))
        print(f"  [EVENT BUS] Subscribed {handler_name} to '{event_type}'")

    def unsubscribe(self, event_type: str, handler: EventHandler):
        """Unsubscribe from an event type"""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(handler)

    def publish(self, event: Event):
        """Publish an event to all subscribers"""
        print(f"  [EVENT BUS] Publishing '{event.event_type}' from {event.source}")
        self.event_history.append(event)

        if event.event_type in self.subscribers:
            for handler in self.subscribers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    print(f"    [ERROR] Handler failed: {e}")

    def get_subscribers_count(self, event_type: str) -> int:
        """Get the number of subscribers for an event type"""
        return len(self.subscribers.get(event_type, []))


# Example event handlers
def user_created_handler(event: Event):
    """Handle a user-created event"""
    user_id = event.data.get("user_id")
    print(f"    [HANDLER] Sending welcome email to user {user_id}")


def user_created_analytics(event: Event):
    """Track user creation in analytics"""
    print(f"    [HANDLER] Recording user creation in analytics")


def order_placed_handler(event: Event):
    """Handle an order-placed event"""
    order_id = event.data.get("order_id")
    print(f"    [HANDLER] Processing payment for order {order_id}")


def order_placed_notification(event: Event):
    """Send notification for order"""
    order_id = event.data.get("order_id")
    print(f"    [HANDLER] Sending order confirmation for {order_id}")


# 4. API VERSIONING
class ApiVersion(Enum):
    """API versions"""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


@dataclass
class ApiResponse:
    """API response"""
    version: ApiVersion
    data: dict
    deprecated: bool = False
    deprecation_message: Optional[str] = None


class VersionedApi:
    """
    API versioning supports multiple versions simultaneously.
    Enables gradual migration without breaking existing clients.
    """

    def __init__(self):
        self.endpoints: dict[ApiVersion, dict[str, Callable]] = {
            ApiVersion.V1: {},
            ApiVersion.V2: {},
            ApiVersion.V3: {}
        }
        self.default_version = ApiVersion.V2

    def register_endpoint(self, version: ApiVersion, path: str, handler: Callable):
        """Register an endpoint for a specific version"""
        self.endpoints[version][path] = handler
        print(f"  [API] Registered {version.value}/{path}")

    def call_endpoint(self, path: str, version: Optional[ApiVersion] = None, **kwargs) -> ApiResponse:
        """Call an API endpoint"""
        version = version or self.default_version

        if path not in self.endpoints[version]:
            raise ValueError(f"Endpoint {version.value}/{path} not found")

        handler = self.endpoints[version][path]
        data = handler(**kwargs)

        # Check if the version is deprecated
        deprecated = version == ApiVersion.V1
        deprecation_msg = "API v1 is deprecated. Please upgrade to v2." if deprecated else None

        return ApiResponse(
            version=version,
            data=data,
            deprecated=deprecated,
            deprecation_message=deprecation_msg
        )

    def list_endpoints(self, version: ApiVersion) -> list[str]:
        """List all endpoints for a version"""
        return list(self.endpoints[version].keys())


# Example API handlers
def get_user_v1(user_id: int) -> dict:
    """V1: Returns basic user info"""
    return {
        "id": user_id,
        "name": f"User {user_id}"
    }


def get_user_v2(user_id: int) -> dict:
    """V2: Returns enhanced user info with email"""
    return {
        "id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com",
        "profile": {"verified": True}
    }


def get_user_v3(user_id: int) -> dict:
    """V3: Returns comprehensive user info"""
    return {
        "user_id": user_id,  # Changed field name
        "display_name": f"User {user_id}",
        "contact": {
            "email": f"user{user_id}@example.com",
            "phone": "+1234567890"
        },
        "profile": {
            "verified": True,
            "created_at": "2024-01-15T10:30:00Z"
        }
    }


# 5. FEATURE FLAGS
class FeatureFlag:
    """A feature flag configuration"""

    def __init__(self, name: str, enabled: bool = False, rollout_percentage: int = 0):
        self.name = name
        self.enabled = enabled
        self.rollout_percentage = rollout_percentage  # 0-100
        self.metadata: dict = {}


class FeatureFlagManager:
    """
    Feature flags enable/disable features at runtime.
    Allows gradual rollouts, A/B testing, and quick rollbacks.
    """

    def __init__(self):
        self.flags: dict[str, FeatureFlag] = {}

    def create_flag(self, name: str, enabled: bool = False, rollout_percentage: int = 0):
        """Create a new feature flag"""
        flag = FeatureFlag(name, enabled, rollout_percentage)
        self.flags[name] = flag
        print(f"  [FEATURE FLAG] Created '{name}' (enabled={enabled}, rollout={rollout_percentage}%)")

    def enable(self, flag_name: str):
        """Enable a feature flag"""
        if flag_name in self.flags:
            self.flags[flag_name].enabled = True
            print(f"  [FEATURE FLAG] Enabled '{flag_name}'")

    def disable(self, flag_name: str):
        """Disable a feature flag"""
        if flag_name in self.flags:
            self.flags[flag_name].enabled = False
            print(f"  [FEATURE FLAG] Disabled '{flag_name}'")

    def set_rollout(self, flag_name: str, percentage: int):
        """Set rollout percentage for gradual rollout"""
        if flag_name in self.flags:
            self.flags[flag_name].rollout_percentage = max(0, min(100, percentage))
            print(f"  [FEATURE FLAG] Set '{flag_name}' rollout to {percentage}%")

    def is_enabled(self, flag_name: str, user_id: Optional[str] = None) -> bool:
        """Check if a feature is enabled"""
        if flag_name not in self.flags:
            return False

        flag = self.flags[flag_name]

        # If explicitly enabled/disabled, use that
        if flag.enabled:
            return True

        # Check rollout percentage
        if flag.rollout_percentage > 0 and user_id:
            # Use user_id hash for a consistent assignment
            user_hash = hash(user_id) % 100
            return user_hash < flag.rollout_percentage

        return False

    def list_flags(self) -> list[dict]:
        """List all feature flags"""
        return [
            {
                "name": flag.name,
                "enabled": flag.enabled,
                "rollout_percentage": flag.rollout_percentage
            }
            for flag in self.flags.values()
        ]


def demonstrate_plugin_architecture():
    """Demonstrate plugin architecture"""
    print("\n=== PLUGIN ARCHITECTURE ===")

    manager = PluginManager()

    # Register plugins
    print("\nRegistering plugins:")
    manager.register_plugin(AuthenticationPlugin())
    manager.register_plugin(LoggingPlugin())
    manager.register_plugin(MetricsPlugin())

    print(f"\nRegistered plugins: {manager.list_plugins()}")

    # Execute plugins
    print("\nExecuting all plugins:")
    context = {"user": "alice", "action": "login"}
    results = manager.execute_all(context)
    for plugin_name, result in results.items():
        print(f"  {plugin_name}: {result}")


def demonstrate_service_registry():
    """Demonstrate service registry"""
    print("\n=== SERVICE REGISTRY ===")

    registry = ServiceRegistry()

    # Register services
    print("\nRegistering services:")
    registry.register(ServiceMetadata(
        service_id="api-1",
        name="api-gateway",
        version="1.0.0",
        host="192.168.1.10",
        port=8080,
        endpoints=["/users", "/orders"]
    ))
    registry.register(ServiceMetadata(
        service_id="api-2",
        name="api-gateway",
        version="1.0.0",
        host="192.168.1.11",
        port=8080,
        endpoints=["/users", "/orders"]
    ))
    registry.register(ServiceMetadata(
        service_id="db-1",
        name="database",
        version="2.5.1",
        host="192.168.1.20",
        port=5432,
        endpoints=["/query"]
    ))

    # Discover services
    print("\nDiscovering 'api-gateway' services:")
    services = registry.discover("api-gateway")
    for service in services:
        print(f"  {service.service_id}: {service.host}:{service.port}")

    # Heartbeat
    print("\nSending heartbeat for api-1:")
    registry.heartbeat("api-1")

    print(f"\nTotal registered services: {len(registry.list_services())}")


def demonstrate_event_driven():
    """Demonstrate event-driven architecture"""
    print("\n=== EVENT-DRIVEN ARCHITECTURE ===")

    event_bus = EventBus()

    # Subscribe handlers
    print("\nSubscribing event handlers:")
    event_bus.subscribe("user.created", user_created_handler)
    event_bus.subscribe("user.created", user_created_analytics)
    event_bus.subscribe("order.placed", order_placed_handler)
    event_bus.subscribe("order.placed", order_placed_notification)

    # Publish events
    print("\nPublishing 'user.created' event:")
    event_bus.publish(Event(
        event_type="user.created",
        data={"user_id": "user_123", "email": "user@example.com"},
        source="user-service"
    ))

    print("\nPublishing 'order.placed' event:")
    event_bus.publish(Event(
        event_type="order.placed",
        data={"order_id": "order_456", "amount": 99.99},
        source="order-service"
    ))

    print(f"\nTotal events published: {len(event_bus.event_history)}")


def demonstrate_api_versioning():
    """Demonstrate API versioning"""
    print("\n=== API VERSIONING ===")

    api = VersionedApi()

    # Register endpoints for different versions
    print("\nRegistering versioned endpoints:")
    api.register_endpoint(ApiVersion.V1, "users", get_user_v1)
    api.register_endpoint(ApiVersion.V2, "users", get_user_v2)
    api.register_endpoint(ApiVersion.V3, "users", get_user_v3)

    # Call different versions
    print("\nCalling v1/users (deprecated):")
    response = api.call_endpoint("users", ApiVersion.V1, user_id=42)
    print(f"  Data: {response.data}")
    if response.deprecated:
        print(f"  Warning: {response.deprecation_message}")

    print("\nCalling v2/users (current):")
    response = api.call_endpoint("users", ApiVersion.V2, user_id=42)
    print(f"  Data: {response.data}")

    print("\nCalling v3/users (latest):")
    response = api.call_endpoint("users", ApiVersion.V3, user_id=42)
    print(f"  Data: {response.data}")


def demonstrate_feature_flags():
    """Demonstrate feature flags"""
    print("\n=== FEATURE FLAGS ===")

    ff_manager = FeatureFlagManager()

    # Create flags
    print("\nCreating feature flags:")
    ff_manager.create_flag("new_ui", enabled=False)
    ff_manager.create_flag("dark_mode", enabled=True)
    ff_manager.create_flag("beta_features", rollout_percentage=25)

    # Check flags
    print("\nChecking feature flags:")
    print(f"  new_ui enabled: {ff_manager.is_enabled('new_ui')}")
    print(f"  dark_mode enabled: {ff_manager.is_enabled('dark_mode')}")

    # Test rollout for different users
    print("\nTesting 25% rollout for 'beta_features':")
    test_users = ["user_1", "user_2", "user_3", "user_4", "user_5", "user_6", "user_7", "user_8"]
    enabled_count = 0
    for user in test_users:
        enabled = ff_manager.is_enabled("beta_features", user)
        if enabled:
            enabled_count += 1
        print(f"  {user}: {'✓ enabled' if enabled else '✗ disabled'}")
    print(f"  Actual rollout: {(enabled_count/len(test_users))*100:.0f}%")

    # Enable feature
    print("\nEnabling 'new_ui' feature:")
    ff_manager.enable("new_ui")
    print(f"  new_ui enabled: {ff_manager.is_enabled('new_ui')}")


def main():
    """Run all demonstrations"""
    print("EXTENSIBILITY PATTERNS IN DISTRIBUTED SYSTEMS")
    print("=" * 60)

    demonstrate_plugin_architecture()
    demonstrate_service_registry()
    demonstrate_event_driven()
    demonstrate_api_versioning()
    demonstrate_feature_flags()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("1. Plugin architecture - Add features without modifying core")
    print("2. Service registry - Dynamic service discovery")
    print("3. Event-driven - Loose coupling through events")
    print("4. API versioning - Support multiple versions simultaneously")
    print("5. Feature flags - Control features at runtime")
    print("\nExtensibility = Flexibility to evolve without breaking existing functionality")


if __name__ == "__main__":
    main()