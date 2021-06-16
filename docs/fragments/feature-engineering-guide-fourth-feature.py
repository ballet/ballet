import numpy as np
from ballet import Feature
from ballet.eng.external import MinMaxScaler
from predict_house_prices.features.contrib.user_jane.feature_impute_lot_frontage import (
    feature as impute_lot_frontage,
)

input = ["Lot Frontage", "Lot Area"]
transformer = [
    (impute_lot_frontage.input, impute_lot_frontage.transformer),
    lambda df: df["Lot Area"] / df["Lot Frontage"] ** 2,
    lambda ser: ser.clip(lower=1.0),
    lambda ser: np.log(ser),
    MinMaxScaler(),
]
name = "Narrowness of the lot"
feature = Feature(input, transformer, name=name)
