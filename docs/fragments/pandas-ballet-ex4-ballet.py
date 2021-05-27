from ballet import Feature
import numpy as np

input = "Lot Frontage"
transformer = [
    np.log,
    lambda ser: ser.fillna(0),
]
feature = Feature(input=input, transformer=transformer)
