import pandas as pd

X_df = pd.read_json(
    '{"Neighborhood":{"0":"NAmes","1":"NAmes","2":"NAmes","3":"NAmes","4":"Gilbert"},"Lot Frontage":{"0":141.0,"1":80.0,"2":null,"3":93.0,"4":74.0},"Lot Area":{"0":31770,"1":11622,"2":14267,"3":11160,"4":13830},"Exterior 1st":{"0":null,"1":"VinylSd","2":"Wd Sdng","3":"BrkFace","4":"VinylSd"},"Garage Area":{"0":528.0,"1":730.0,"2":312.0,"3":522.0,"4":null},"1st Flr SF":{"0":1656,"1":896,"2":1329,"3":2110,"4":928}}'
)
y_df = pd.read_json(
    '{"0":215000,"1":105000,"2":172000,"3":244000,"4":189900}', typ="series"
)
