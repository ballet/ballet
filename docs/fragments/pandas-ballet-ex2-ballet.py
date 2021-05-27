from ballet import Feature

input = ["Lot Frontage", "Lot Area"]
transformer = lambda df: df.fillna(0)
feature = Feature(input=input, transformer=transformer)
