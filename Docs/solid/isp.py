"""
Interface Separation Principle
"""

from abc import ABC, abstractmethod

# ISP applied: Separate interfaces for Printer and Scanner
class Printer(ABC):
    @abstractmethod
    def print(self, content: str) -> None:
        pass

class Scanner(ABC):
    @abstractmethod
    def scan(self) -> str:
        pass

# Implementation of Printer
class LaserPrinter(Printer):
    def print(self, content: str) -> None:
        print(f"Printing: {content}")

# Implementation of Scanner
class DocumentScanner(Scanner):
    def scan(self) -> str:
        return "Scanned Document Content"

# Specific class only uses Printer
class User:
    def __init__(self, printer: Printer):
        self.printer = printer

    def print_document(self, content: str):
        self.printer.print(content)

# Specific class only uses Scanner
class Archiver:
    def __init__(self, scanner: Scanner):
        self.scanner = scanner

    def archive_document(self):
        content = self.scanner.scan()
        print(f"Archived content: {content}")
