class Counter():
    def __init__(self, name):
        self.name = name
        self.value = 0
        
    def increment(self, n=1):
        self.value += n
    
    def reset(self):
        self.value = 0
    
    @property
    def is_zero(self):
        return self.value == 0
    
    @property
    def info(self):
        return f"{self.name}: {self.value}"
    

class CappedCounter(Counter):
    def __init__(self, name, max_value):
        self._validate_max(max_value)
        self.max_value = max_value
        super().__init__(name)
        
    @staticmethod
    def _validate_max(x):
        if not isinstance(x, int) or x < 0:
            raise ValueError("max_value must be a non-negative integer")
        return x
    
    @classmethod
    def with_default_max(cls, name):
        return cls(name, max_value=5)
    
    def increment(self, n=1):
        n = CappedCounter.ensure_nonnegative(n)
        super().increment(n)
        if self.value > self.max_value:
            self.value = self.max_value
    
    @property
    def remaining(self):
        r = self.max_value - self.value
        return r if r > 0 else 0
    
    @staticmethod
    def ensure_nonnegative(n):
        return n if n >= 0 else 0
    
    @classmethod
    def from_config(cls, cfg: dict):
        name = cfg.get("name", None)
        if name is None:
            raise ValueError("Config must include 'name'")
        max_value = cfg.get("max_value", 8)
        if not isinstance(max_value, int) or max_value < 0:
            raise ValueError("'max_value' must be a non-negative integer")
        cls._validate_max(max_value)
        return cls(name, max_value)
    
    def reset(self, to_max=False):
        if to_max:
            self.value = self.max_value
        else:
            super().reset()

class CounterManager():
    def __init__(self):
        self.store = {}
        
    def add_counter(self, obj):
        if obj.name in self.store:
            raise ValueError(f"Counter with name {obj.name} already exists.")
        self.store[obj.name] = obj # 这里的 obj 是值/item
        
    @property
    def total(self):
        return sum(c.value for c in self.store.values()) #  这里的self.store.values() 是所有的 item, 也就是所有的 obj
    
    def bump(self, name, n=1):
        if name not in self.store:
            raise KeyError(f"No counter found with name {name}")
        self.store[name].increment(n)
        
    def __len__(self):
        return len(self.store)
    
    def __getitem__(self, key):
        if isinstance(key, str): 
            return self.store[key]
        elif isinstance(key, int):
            if key < 0 or key >= len(self.store):
                raise IndexError("Index out of range")
            sorted_keys = sorted(self.store.keys())
            return self.store[sorted_keys[key]]
        else:
            raise TypeError("Key must be a string or an integer")
    def __contains__(self, name: str) -> bool:
        return name in self.store

    def __iter__(self):
        for name in sorted(self.store.keys()):
            yield self.store[name]

    @property
    def names(self):
        return list(self.store.keys())
    

def main():
    c = CappedCounter.from_config({"name": "ok", "max_value": -3})
    print(c.info)   # 期望：ok: 0



if __name__ == "__main__":
    main()