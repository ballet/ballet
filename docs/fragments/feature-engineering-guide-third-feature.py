from ballet import Feature
from ballet.eng import NullFiller

input = ["Total Bsmt SF", "1st Flr SF", "2nd Flr SF"]
transformer = [
    lambda df: df.sum(axis=1),
    NullFiller()
]
name = "Total Area"
feature = Feature(input=input, transformer=transformer, name=name)
