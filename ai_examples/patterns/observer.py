class Subject:
    def __init__(self):
        self._observers = []  # List to keep track of registered observers

    def register_observer(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer):
        if observer in self._observers:
            self._observers.remove(observer)

    def notify_observers(self):
        for observer in self._observers:
            observer.update(self)


class Observer:
    def update(self, subject):
        raise NotImplementedError("Subclasses should implement this!")


# Example usage:
class ConcreteSubject(Subject):
    def __init__(self):
        super().__init__()
        self._state = 0  # Example state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self.notify_observers()  # Notify observers of state change


class ConcreteObserver(Observer):
    def update(self, subject):
        print(f"Observer notified. Subject state is now {subject.state}")


# Instantiate the subject and observers
subject = ConcreteSubject()
observer1 = ConcreteObserver()
observer2 = ConcreteObserver()

# Register observers
subject.register_observer(observer1)
subject.register_observer(observer2)

# Change subject state
subject.state = 42  # This triggers notifications to observers
