
class Config:
    def __init__(self, config=None):
        self.data = config if config is not None else {}

    def set(self, key, value, description=None):
        self.data[key] = {"value": value, "description": description}

    def __getattr__(self, item):
        return self.data[item]['value']
