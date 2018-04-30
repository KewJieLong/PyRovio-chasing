class Node(object):
    def __init__(self, element):
        self.element = element
        self.tail = None

    def set_element(self, e):
        self.element = e
