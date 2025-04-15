from typing import TypeAlias, SupportsFloat, SupportsInt

from numpy.typing import NDArray

Scalar: TypeAlias = SupportsFloat | SupportsInt | bool
Numeric: TypeAlias = NDArray[Scalar] | Scalar
