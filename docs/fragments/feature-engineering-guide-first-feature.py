from ballet import Feature
from ballet.eng import IdentityTransformer

input = 'Lot Area'
transformer = IdentityTransformer()
feature = Feature(input=input, transformer=transformer)
