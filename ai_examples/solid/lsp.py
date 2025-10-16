"""
Liskov Substitution Principle
"""

class Bird:
    def eat(self):
        print("This bird is eating.")

class FlyingBird(Bird):
    def fly(self):
        print("This bird can fly!")

class Sparrow(FlyingBird):
    def chirp(self):
        print("Chirp chirp!")

class Penguin(Bird):
    def swim(self):
        print("This bird can swim!")

# Adherence to Liskov Substitution Principle
def make_bird_fly(bird: FlyingBird):
    bird.fly()

sparrow = Sparrow()
penguin = Penguin()

# Example usage demonstrating LSP
make_bird_fly(sparrow)  # Works fine
# make_bird_fly(penguin)  # This would cause an error, illustrating design adherence
