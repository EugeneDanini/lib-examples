"""
Single responsibility principle
"""

class Employee:
    def __init__(self, name: str, position: str, salary: float):
        self.name = name
        self.position = position
        self.salary = salary


class EmployeePrinter:
    def print_details(self, employee: Employee):
        print(f"Name: {employee.name}")
        print(f"Position: {employee.position}")
        print(f"Salary: {employee.salary}")


# Example usage:
if __name__ == "__main__":
    employee = Employee("Alice", "Developer", 75000)
    printer = EmployeePrinter()
    printer.print_details(employee)
