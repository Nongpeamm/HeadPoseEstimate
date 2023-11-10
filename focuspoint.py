class FocusPoint:
    def __init__(self, ):
        self.x, self.y = None, None

    def set_focus(self, x, y):
        self.x = x
        self.y = y
    
    def get_focus(self):
        return self.x, self.y