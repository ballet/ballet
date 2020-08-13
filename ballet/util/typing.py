from typing import Callable, Collection, TypeVar, Union

import ballet.eng  # noqa
from ballet.compat import PathLike

T = TypeVar('T')
OneOrMore = Union[T, Collection[T]]
Pathy = Union[str, PathLike]
TransformerLike = Union[Callable, 'ballet.eng.BaseTransformer', None]
