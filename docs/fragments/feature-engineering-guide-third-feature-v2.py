from ballet import Feature
from ballet.eng import NullFiller
from ballet.eng.sklearn import StandardScaler

input = ["Total Bsmt Sf", "1st Flr SF", "2nd Flr SF"]
transformer = [
    NullFiller(),
    StandardScaler(),
]
feature = Feature(input, transformer)
