import inspect
from dataclasses import dataclass
from typing import Any, Callable

from tenacity import (
    AsyncRetrying,
    retry,
    stop_after_attempt,
    stop_after_delay,
    wait_fixed,
)


@dataclass
class Tool:
    """
    A class representing a tool with a name, description, function, arguments, and outputs.

    Parameters
    ----------
    name : str
        The name of the tool.
    description : str
        A description of what the tool does.
    func : Callable
        The function that implements the tool's functionality.
    arguments : list[tuple[str, str]]
        List of tuples containing argument names and their types.
    outputs : str
        Description of the function's return type.
    """

    name: str
    description: str
    func: Callable
    arguments: list[tuple[str, str]]
    outputs: str

    def to_string(self) -> str:
        """
        Convert the Tool object to a string representation.

        Returns
        -------
        str
            A formatted string containing the tool's details.
        """
        args_str = ", ".join([f"{name}: {_type}" for name, _type in self.arguments])
        return (
            f"Tool Name: {self.name}, Description: {self.description}, Arguments: {args_str}",
            f"Outputs: {self.outputs}",
        )  # type: ignore

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the function with the provided arguments.

        Parameters
        ----------
        *args : Any
            Positional arguments to pass to the function.
        **kwargs : Any
            Keyword arguments to pass to the function.

        Returns
        -------
        Any
            The result of calling the function.
        """
        return self.func(*args, **kwargs)


def tool(func: Callable) -> Tool:
    """
    Decorator to create a Tool object from a function.

    Parameters
    ----------
    func : Callable
        The function to convert into a Tool.

    Returns
    -------
    Tool
        A Tool object wrapping the provided function.
    """
    signature = inspect.signature(func)
    # Extract the name and annotation
    arguments: list[tuple[str, str]] = []
    for param in signature.parameters.values():
        annotation_name: str = (
            param.annotation.__name__
            if hasattr(param.annotation, "__name__")
            else str(param.annotation)
        )
        arguments.append((param.name, annotation_name))

    # Determine the return annotation
    return_annotation = signature.return_annotation
    if return_annotation is inspect._empty:  # noqa: SLF001
        outputs: str = "No return annotation"
    else:
        outputs = (
            return_annotation.__name__
            if hasattr(return_annotation, "__name__")
            else str(return_annotation)
        )
    # Function docstring
    docstring = func.__doc__ or "No description provided."

    # Create the Tool object
    return Tool(
        name=func.__name__,
        description=docstring,
        func=func,
        arguments=arguments,
        outputs=outputs,
    )


def async_retrying_with_print() -> AsyncRetrying:
    """
    Creates and returns an AsyncRetrying instance with custom retry configuration.

    The function configures retry behavior with fixed wait time, maximum attempts,
    maximum delay, and custom callback functions for retry events.

    Returns
    -------
    AsyncRetrying
        Configured AsyncRetrying instance with the following settings:
        - Fixed wait time of 1 second between retries
        - Stops after 5 attempts or 30 seconds delay
        - Prints before and after each retry attempt
        - Prints final failure information if all retries fail
    """
    return AsyncRetrying(
        wait=wait_fixed(1),
        stop=(stop_after_attempt(5) | stop_after_delay(180)),
        before=lambda _: print("before:", _),
        after=lambda _: print("after:", _),
        # Add custom statistics
        retry_error_callback=lambda state: print(
            f"Final failure after {state.attempt_number} attempts: {{state.outcome.exception()}}"  # type: ignore
        ),
    )


def simple_retry(attempts: int = 5, delay: int = 1, timeout: int = 30) -> Callable[..., Any]:
    """
    A clean, simple retry decorator with logging.

    Parameters
    ----------
    attempts : int, optional
        Maximum number of retry attempts, by default 5
    delay : int, optional
        Fixed delay between attempts in seconds, by default 1
    timeout : int, optional
        Maximum total time in seconds before stopping retries, by default 30

    Returns
    -------
    callable
        A configured retry decorator that implements the specified retry behavior
        with logging of attempts and failures.
    """
    return retry(
        wait=wait_fixed(delay),
        stop=(stop_after_attempt(attempts) | stop_after_delay(timeout)),
        before=lambda retry_state: print(f"Attempt {retry_state.attempt_number}..."),
        after=lambda retry_state: print(f"Completed attempt {retry_state.attempt_number}"),
        retry_error_callback=lambda retry_state: print(
            f"Failed after {retry_state.attempt_number} attempts: {retry_state.outcome.exception()}"  # type: ignore
        ),
    )