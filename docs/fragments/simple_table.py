import pandas as pd
X_df = pd.DataFrame(data={
    'Lot Frontage': [141, None, 81, 93, 74],
    'Lot Area': [31770, 11622, 14267, None, 13830],
    'Exterior 1st': ['BrkFace', 'VinylSd', 'Wd Sdng', 'BrkFace', 'VinylSd'],
})
y_df = pd.Series(data=[215000, 105000, 172000, 244000, 189900], name='Sale Price')
