from typing import Literal, TypedDict, Any

class Boundary_Condition(TypedDict):
    type: tuple[Literal['constant', 'symmetry'], ...]
    value: tuple[Any, ...]