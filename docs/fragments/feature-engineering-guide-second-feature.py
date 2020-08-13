from ballet import Feature
from ballet.eng.sklearn import SimpleImputer

input = ["Lot Frontage"]
transformer = SimpleImputer(strategy="mean")
name = "Imputed Lot Frontage"
feature = Feature(input=input, transformer=transformer, name=name)
