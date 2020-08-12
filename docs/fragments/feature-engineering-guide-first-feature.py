from ballet import Feature
from ballet.eng.misc import IdentityTransformer

input = 'Lot Area'
transformer = IdentityTransformer()
feature = Feature(input=input, transformer=transformer)
