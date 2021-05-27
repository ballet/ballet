from ballet import Feature

input = "Lot Frontage"
transformer = lambda ser: ser.fillna(0)
feature = Feature(input=input, transformer=transformer)
