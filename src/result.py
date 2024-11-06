from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar


class Option[T](ABC):
    @abstractmethod
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    @abstractmethod
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool:
        return not (self == other)

    @abstractmethod
    def and_[U](self, other: "Option[U]") -> "Option[U]": ...
    @abstractmethod
    def and_then[U](self, f: Callable[[T], "Option[U]"]) -> "Option[U]": ...
    @abstractmethod
    def expect(self, message: str) -> T: ...
    @abstractmethod
    def filter(self, f: Callable[[T], bool]) -> "Option[T]": ...
    @abstractmethod
    def get_or_insert(self, value: T) -> T: ...
    @abstractmethod
    def get_or_insert_with(self, f: Callable[[], T]) -> T: ...


class Some[T](Option[T]):
    def __init__(self, value: T) -> None:
        self._value = value

    def __repr__(self) -> str:
        return f"Some({self._value.__repr__()})"

    def __hash__(self) -> int:
        return hash(("Some", self._value))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Some) and self._value == other._value

    def and_[U](self, other: "Option[U]") -> "Option[U]":
        return other

    def and_then[U](self, f: Callable[[T], "Option[U]"]) -> "Option[U]":
        return f(self._value)

    def expect(self, message: str) -> T:
        return self._value

    def filter(self, f: Callable[[T], bool]) -> Option[T]:
        if f(self._value):
            return self
        return Null()

    def get_or_insert(self, value: T) -> T:
        return self._value

    def get_or_insert_with(self, f: Callable[[], T]) -> T:
        return self._value


class Null(Option):
    def __repr__(self) -> str:
        return "Null"

    def __hash__(self) -> int:
        return hash(None)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Null) or other is None

    def and_[U](self, other: Option[U]) -> Option[U]:
        return Null()

    def and_then[U](self, f: Callable[[Any], Option[U]]) -> Option[U]:
        return Null()

    def expect(self, message: str) -> Any:
        raise ValueError(f"{message}")

    def filter(self, f: Callable[[Any], bool]) -> Option:
        return Null()

    def get_or_insert(self, value: Any) -> Any:
        return

    def get_or_insert_with(self, f: Callable[[], Any]) -> Any:
        return


class Result[T, E](ABC):
    @abstractmethod
    def __repr__(self) -> str: ...
    @abstractmethod
    def __hash__(self) -> int: ...
    @abstractmethod
    def __eq__(self, other: object) -> bool: ...
    def __nq__(self, other: object) -> bool:
        return not (self == other)

    @abstractmethod
    def and_[U](self, other: "Result[U, E]") -> "Result[U, E]": ...
    @abstractmethod
    def and_then[U](self, f: Callable[[T], "Result[U, E]"]) -> "Result[U, E]": ...
    @abstractmethod
    def err(self) -> E | None: ...
    @abstractmethod
    def expect(self, message: str) -> T: ...
    @abstractmethod
    def expect_err(self, message: str) -> E: ...
    @abstractmethod
    def inspect(self, f: Callable[[T], None]) -> "Result[T, E]": ...
    @abstractmethod
    def inspect_err(self, f: Callable[[E], None]) -> "Result[T, E]": ...
    @abstractmethod
    def is_err(self) -> bool: ...
    @abstractmethod
    def is_err_and(self, f: Callable[[E], bool]) -> bool: ...
    @abstractmethod
    def is_ok(self) -> bool: ...
    @abstractmethod
    def is_ok_and(self, f: Callable[[T], bool]) -> bool: ...
    @abstractmethod
    def map[U](self, f: Callable[[T], U]) -> "Result[U, E]": ...
    @abstractmethod
    def map_err[F](self, f: Callable[[E], F]) -> "Result[T, F]": ...
    @abstractmethod
    def map_or[U](self, default: U, f: Callable[[T], U]) -> U: ...
    @abstractmethod
    def map_or_else[U](self, default: Callable[[E], U], f: Callable[[T], U]) -> U: ...
    @abstractmethod
    def ok(self) -> T | None: ...
    @abstractmethod
    def or_[F](self, other: "Result[T, F]") -> "Result[T, F]": ...
    @abstractmethod
    def or_else[F](self, f: Callable[[E], "Result[T, F]"]) -> "Result[T, F]": ...
    @abstractmethod
    def unwrap(self) -> T: ...
    @abstractmethod
    def unwrap_err(self) -> E: ...
    @abstractmethod
    def unwrap_or(self, default: T) -> T: ...
    @abstractmethod
    def unwrap_or_else(self, f: Callable[[E], T]) -> T: ...


class Ok[T, E](Result[T, E]):
    __match_args__ = ("_value",)

    def __init__(self, value: T) -> None:
        self._value = value

    def __repr__(self) -> str:
        return f"Ok({self._value.__repr__()})"

    def __hash__(self) -> int:
        return hash(("Ok", self._value))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ok) and self._value == other._value

    def and_[U](self, other: Result[U, E]) -> Result[U, E]:
        return other

    def and_then[U](self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return f(self._value)

    def err(self) -> E | None:
        return None

    def expect(self, message: str) -> T:
        return self._value

    def expect_err(self, message: str) -> E:
        raise ValueError(f"{message}: {self._value.__repr__()}")

    def inspect(self, f: Callable[[T], None]) -> Result[T, E]:
        f(self._value)
        return self

    def inspect_err(self, f: Callable[[E], None]) -> Result[T, E]:
        return self

    def is_err(self) -> bool:
        return False

    def is_ok(self) -> bool:
        return True

    def is_err_and(self, f: Callable[[E], bool]) -> bool:
        return False

    def is_ok_and(self, f: Callable[[T], bool]) -> bool:
        return f(self._value)

    def map[U](self, f: Callable[[T], U]) -> Result[U, E]:
        return Ok(f(self._value))

    def map_err[F](self, f: Callable[[E], F]) -> Result[T, F]:
        return Ok(self._value)

    def map_or[U](self, default: U, f: Callable[[T], U]) -> U:
        return f(self._value)

    def map_or_else[U](self, default: Callable[[E], U], f: Callable[[T], U]) -> U:
        return f(self._value)

    def ok(self) -> T | None:
        return self._value

    def or_[F](self, other: Result[T, F]) -> Result[T, F]:
        return Ok(self._value)

    def or_else[F](self, f: Callable[[E], Result[T, F]]) -> Result[T, F]:
        return Ok(self._value)

    def unwrap(self) -> T:
        return self._value

    def unwrap_err(self) -> E:
        raise ValueError("ERROR: Called unwrap_err on an Ok value.")

    def unwrap_or(self, default: T) -> T:
        return self._value

    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        return self._value


class Err[T, E](Result[T, E]):
    __match_args__ = ("_error",)

    def __init__(self, error: E) -> None:
        self._error = error

    def __repr__(self) -> str:
        return f"Err({self._error.__repr__()})"

    def __hash__(self) -> int:
        return hash(("Ok", self._error))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Err) and self._error == other._error

    def and_[U](self, other: Result[U, E]) -> Result[U, E]:
        return Err(self._error)

    def and_then[U](self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return Err(self._error)

    def err(self) -> E | None:
        return self._error

    def expect(self, message: str) -> T:
        raise ValueError(f"{message}: {self._error.__repr__()}")

    def expect_err(self, message: str) -> E:
        return self._error

    def inspect(self, f: Callable[[T], None]) -> Result[T, E]:
        return self

    def inspect_err(self, f: Callable[[E], None]) -> Result[T, E]:
        f(self._error)
        return self

    def is_err(self) -> bool:
        return True

    def is_ok(self) -> bool:
        return False

    def is_err_and(self, f: Callable[[E], bool]) -> bool:
        return f(self._error)

    def is_ok_and(self, f: Callable[[T], bool]) -> bool:
        return False

    def map[U](self, f: Callable[[T], U]) -> Result[U, E]:
        return Err(self._error)

    def map_err[F](self, f: Callable[[E], F]) -> Result[T, F]:
        return Err(f(self._error))

    def map_or[U](self, default: U, f: Callable[[T], U]) -> U:
        return default

    def map_or_else[U](self, default: Callable[[E], U], f: Callable[[T], U]) -> U:
        return default(self._error)

    def ok(self) -> T | None:
        return None

    def or_[F](self, other: Result[T, F]) -> Result[T, F]:
        return other

    def or_else[F](self, f: Callable[[E], Result[T, F]]) -> Result[T, F]:
        return f(self._error)

    def unwrap(self) -> T:
        raise ValueError("ERROR: Attempted to unwrap an Err.")

    def unwrap_err(self) -> E:
        return self._error

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        return f(self._error)


def divide(a: int, b: int) -> Result[int, str]:
    if b == 0:
        return Err("Cannot divide by 0.")
    return Ok(a // b)


def find_first[T](item: T, items: list[T]) -> Option[int]:
    return Null()


if __name__ == "__main__":
    print(divide(1, 2))
    print(divide(1, 0))
    match divide(1, 2):
        case Ok(value):
            print(f"Success: {value}")
        case Err(error):
            print(f"Error: {error}")
