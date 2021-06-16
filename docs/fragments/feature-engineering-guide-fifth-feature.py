from ballet import Feature
from ballet.eng import GroupwiseTransformer
from ballet.eng.external import SimpleImputer, OneHotEncoder

input = ["Neighborhood", "Exterior 1st"]
transformer = [
    GroupwiseTransformer(
        SimpleImputer(strategy="most_frequent"),
        groupby_kwargs={"by": "Neighborhood"},
        column_selection=["Exterior 1st"],
    ),
    OneHotEncoder(),
]
name = "Cleaned and encoded exterior covering"
feature = Feature(input, transformer, name=name)
