from ballet import Feature
from sklearn.impute import SimpleImputer

input = 'Lot Frontage'
transformer = SimpleImputer(strategy='mean')
feature = Feature(input=input, transformer=transformer)
