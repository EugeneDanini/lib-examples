"""
Dependency Injection (DI) pattern example

Goal
- Decouple components by depending on abstractions, not concrete implementations.
- Make swapping implementations easy (e.g., for testing, environment changes).

This file demonstrates three common injection styles:
- Constructor injection (preferred): pass dependencies via __init__
- Setter injection: assign dependencies after construction (optional)
- Provider/factory injection: pass a callable that produces dependencies lazily

No external libraries used; a tiny, explicit Injector is included to show wiring.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Type, TypeVar, Any


# 1) Define abstractions (interfaces)
class Logger(ABC):
    @abstractmethod
    def info(self, msg: str) -> None: ...

    @abstractmethod
    def error(self, msg: str) -> None: ...


# 2) Provide concrete implementations
class ConsoleLogger(Logger):
    def info(self, msg: str) -> None:
        print(f"[INFO] {msg}")

    def error(self, msg: str) -> None:
        print(f"[ERROR] {msg}")


class SilentLogger(Logger):
    """A no-op logger useful for tests."""
    def info(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass


# 3) Consumers depend on abstractions, not concretions
@dataclass
class UserRepository:
    logger: Logger  # constructor injection

    def save(self, user: Dict[str, Any]) -> None:
        self.logger.info(f"Saving user {user['id']}")
        # Imagine persistence here
        # If something goes wrong:
        # self.logger.error("Failed to save user")


@dataclass
class UserService:
    repo: UserRepository  # constructor injection
    _audit_log: Logger | None = None  # optional setter injection

    # Setter injection example: allows replacing audit logger independently
    def set_audit_logger(self, logger: Logger) -> None:
        self._audit_log = logger

    def create_user(self, user_id: str) -> Dict[str, Any]:
        user = {"id": user_id}
        self.repo.save(user)
        if self._audit_log:
            self._audit_log.info(f"User created: {user_id}")
        return user


# 4) A tiny, explicit injector/wiring helper (for illustration only)
T = TypeVar("T")


class Injector:
    """Minimalistic dependency injector.

    - register(type, provider): provider is a callable that returns an instance
    - get(type): resolves the instance using the provider

    This is intentionally very small to keep the example simple.
    """

    def __init__(self) -> None:
        self._providers: Dict[Type[Any], Callable[[Injector], Any]] = {}

    def register(self, key: Type[T], provider: Callable[[Injector], T]) -> None:
        self._providers[key] = provider

    def get(self, key: Type[T]) -> T:
        try:
            provider = self._providers[key]
        except KeyError as e:
            raise KeyError(f"No provider registered for {key!r}") from e
        return provider(self)


# 5) Example composition root (where wiring happens)
def build_app(use_silent_logging: bool = False) -> UserService:
    injector = Injector()

    # Choose a concrete logger based on environment/config
    if use_silent_logging:
        injector.register(Logger, lambda i: SilentLogger())
    else:
        injector.register(Logger, lambda i: ConsoleLogger())

    # Repositories and services depend on abstractions and are wired by the injector
    injector.register(UserRepository, lambda i: UserRepository(logger=i.get(Logger)))
    injector.register(UserService, lambda i: UserService(repo=i.get(UserRepository)))

    # Resolve the root service
    service = injector.get(UserService)

    # Demonstrate setter injection for an audit logger (can differ from main logger)
    service.set_audit_logger(injector.get(Logger))
    return service


# 6) Simple demonstration
if __name__ == "__main__":
    print("-- With ConsoleLogger --")
    app = build_app(use_silent_logging=False)
    app.create_user("u-123")

    print("\n-- With SilentLogger (e.g., tests) --")
    test_app = build_app(use_silent_logging=True)
    test_app.create_user("u-456")
