from ballet import Feature
from ballet.eng.external import SimpleImputer

input = "Lot Frontage"
transformer = SimpleImputer(strategy="mean")
name = "Imputed Lot Frontage"
feature = Feature(input, transformer, name=name)
