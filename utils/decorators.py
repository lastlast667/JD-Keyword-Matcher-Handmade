from functools import wraps
from config.settings import DATA_DIR
from functools import lru_cache

@lru_cache(maxsize=None)    # 缓存装饰器
def load_words_from(filename: str) -> list[str]:
    """
    加载词表装饰器：
    加载停用词表和自定义词表
    :param filename: 词表文件名
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            words_file = DATA_DIR / filename # 词表路径

            words = []
            with open(words_file, 'r', encoding='utf-8') as f:
                for line in f:
                    words.append(line.strip())
            return func(words, *args, **kwargs)
        return wrapper
    return decorator


