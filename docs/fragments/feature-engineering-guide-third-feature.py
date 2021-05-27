from ballet import Feature
from ballet.eng.external import SimpleImputer

input = ["Lot Area", "Garage Area", "1st Flr SF"]
transformer = [
    lambda df: df["Lot Area"] - df["Garage Area"] - df["1st Flr SF"],
    SimpleImputer(strategy="median"),
]
name = "Yard Area"
feature = Feature(input=input, transformer=transformer, name=name)
