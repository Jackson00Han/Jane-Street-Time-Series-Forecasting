from functools import wraps

def log_calls(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[log] calling {fn.__name__} with args={args}, kwargs={kwargs}")
        out = fn(*args, **kwargs)
        print(f"args: {args}, kwargs: {kwargs}")
        print(f"[log] {fn.__name__} returned {out}")
        return out
    return wrapper

@log_calls
def add(a, b):
    return a + b

print(add(2, 5))  # 期望打印两行日志，然后输出 7
