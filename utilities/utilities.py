import inspect
from dataclasses import dataclass
from typing import Any, Callable


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
        )

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
