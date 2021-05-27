import numpy as np

(X_df["Lot Frontage"].apply(np.log).fillna(0))
