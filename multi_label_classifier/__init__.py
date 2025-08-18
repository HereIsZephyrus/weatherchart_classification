"""multi-label classifier package"""

__all__ = ["core", "preprocess"]

# 延迟导入子包，仅在外部代码实际引用时才加载
def __getattr__(name):
    if name == "core":
        from . import core
        return core
    elif name == "preprocess":
        from . import preprocess
        return preprocess
    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}")
