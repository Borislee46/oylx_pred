from collections import OrderedDict
from typing import Generic, Iterable, Iterator, MutableMapping, Optional, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int = 256):
        if capacity <= 0:
            raise ValueError("LRU 缓存容量必须为正数")
        self._capacity = capacity
        self._data: MutableMapping[K, V] = OrderedDict()

    def get(self, key: K) -> Optional[V]:
        if key not in self._data:
            return None
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def put(self, key: K, value: V) -> None:
        if key in self._data:
            self._data.pop(key)
        elif len(self._data) >= self._capacity:
            self._data.popitem()
        self._data[key] = value

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def items(self) -> Iterator[Tuple[K, V]]:
        return iter(self._data.items())


def build_school_key(university: str, major: str) -> str:
    return f"{university}|{major}"


def build_selection_key(background_major: str, schools: Iterable[dict[str, str]]) -> str:
    school_keys = sorted(
        build_school_key(s.get("university", ""), s.get("major", "")) for s in schools
    )
    return f"{background_major}|{'|'.join(school_keys)}"


def build_school_set_key(schools: Iterable[dict[str, str]]) -> str:
    school_keys = sorted(
        build_school_key(s.get("university", ""), s.get("major", "")) for s in schools
    )
    return "|".join(school_keys)
