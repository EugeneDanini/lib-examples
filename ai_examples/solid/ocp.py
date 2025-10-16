"""
Open-Closed Principle

This example demonstrates the Open-Closed Principle, which states that a class should be open for extension but closed for modification.
"""

from abc import ABC, abstractmethod

# Base class/interface for all shapes
class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

# Circle class extending the base Shape interface
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius * self.radius

# Rectangle class extending the base Shape interface
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

# Demonstration of extension:
shapes = [Circle(10), Rectangle(5, 4)]

for shape in shapes:
    print(f"Area: {shape.area()}")
