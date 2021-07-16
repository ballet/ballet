from ballet import Feature
from ballet.eng import BaseTransformer

input = "Exterior 1st"


class LongestStringValue(BaseTransformer):
    def fit(self, X, y=None):
        self.longest_string_length_ = X.str.len().max()
        return self

    def transform(self, X):
        return X.str.len() >= self.longest_string_length_


transformer = LongestStringValue()
feature = Feature(input, transformer)
