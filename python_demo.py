# 写一个简单的计数器类
from functools import wraps

def log_calls(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[log] calling {fn.__name__} args={args}, kwargs={kwargs}")
        out = fn(*args, **kwargs)
        print(f"[log] {fn.__name__} returned {out}")
        return out
    return wrapper


class Counter():
    def __init__(self, name): # 这个函数的arguments: 只有name
        self.name = name
        self.value = 0 # 这个函数的attributes: name, value. 注意 value 不是参数（arguments）! 它只是类的一个属性（attribute）但不是参数。
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
    
    

# 目标：继承你现有的 Counter，新增“上限”逻辑，并在 increment 里用 super() 复用父类实现。

class CappedCounter(Counter):
    def __init__(self, name, max_value):
        self.max_value = max_value
        super().__init__(name)
        print("[CappedCounter.__init__] done, value =", self.value)
        
    @classmethod
    def with_default_max(cls, name):
        print("[with_default_max] cls =", cls.__name__)
        return cls(name, max_value=5)
    
    @staticmethod
    def ensure_nonnegative(n):
        return n if n >= 0 else 0
    
    @log_calls
    def increment(self, n=1):
        print("[increment] before ensure:", n)
        n = CappedCounter.ensure_nonnegative(n)

        super().increment(n)
        if self.value > self.max_value:
            self.value = self.max_value
        print("[increment] after:", self.value)
        
    @property
    def remaining(self):
        r = self.max_value - self.value
        return r if r > 0 else 0
    
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
            raise ValueError(f"Counter named {obj.name} already exists!")
        self.store[obj.name] = obj
    @property
    def total(self):
        return sum(c.value for c in self.store.values())
    
    def bump(self, name: str, n: int = 1):
        if name not in self.store:
            raise KeyError(f"Unknown counter: {name}")
        self.store[name].increment(n)

    @property
    def names(self):
        return list(self.store.keys())


def main():
    m = CounterManager()
    m.add_counter(Counter("a"))
    m.add_counter(CappedCounter("b", max_value=3))
    
    m.store["a"].increment(2)
    m.store["b"].increment(10)
    print("Total:", m.total)
    
    m.bump("a", 3)             # a 由 2 -> 5
    print("Names:", m.names)   # 例如 ['a','b']
    print("Total:", m.total)   # 预期 8  (a=5, b=3)

    
    
if __name__ == "__main__":
    main()