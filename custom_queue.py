class CustomQueue:
    def __init__(self):
        self.queue = []  # List to hold queue elements

    def enqueue(self, item):
        """Add an item to the end of the queue."""
        self.queue.append(item)

    def dequeue(self):
        """Remove and return the item at the front of the queue."""
        if not self.queue:
            raise IndexError("dequeue from an empty queue")
        return self.queue.pop(0)  # Remove the first item

    def delete(self, item):
        """Remove an item from anywhere in the queue."""
        try:
            self.queue.remove(item)  # This raises a ValueError if item not found
        except ValueError:
            print(f"Item '{item}' not found in the queue.")

    def peek(self):
        """Return the item at the front of the queue without removing it."""
        if not self.queue:
            raise IndexError("peek from an empty queue")
        return self.queue[0]

    def is_empty(self):
        """Check if the queue is empty."""
        return len(self.queue) == 0

    def __str__(self):
        """Return a string representation of the queue."""
        return str(self.queue)
