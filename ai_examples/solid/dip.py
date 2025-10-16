"""
Dependency Inversion Principle
"""

from abc import ABC, abstractmethod


class MessageService(ABC):
    @abstractmethod
    def send(self, message: str) -> None:
        pass


class EmailService(MessageService):
    def send(self, message: str) -> None:
        print(f"Email sent with message: {message}")


class MessageSender:
    def __init__(self, service: MessageService) -> None:
        self.service = service

    def send_message(self, message: str) -> None:
        self.service.send(message)


# Example usage
if __name__ == "__main__":
    email_service = EmailService()
    sender = MessageSender(email_service)
    sender.send_message("Hello, Dependency Inversion Principle!")
