from typing import Callable, List, TypeVar, Union

import ballet.eng  # noqa
from ballet.compat import PathLike

T = TypeVar('T')
OneOrMore = Union[T, List[T]]
Pathy = Union[str, PathLike]
TransformerLike = Union[Callable, 'ballet.eng.BaseTransformer', None]
FeatureInputType = Union[OneOrMore[str], Callable[..., OneOrMore[str]]]
FeatureTransformerType = OneOrMore[TransformerLike]
