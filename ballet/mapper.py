import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ballet.feature import Feature, FeatureOutput

__all__ = ['DependentDataFrameMapper']

class DependentDataFrameMapper(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        if not isinstance(features, list):
            features = [features]
        self.features = features
        (sorted_feature_nodes, nodes_by_feature_id) = _construct_feature_DAG(features)
        self.sorted_feature_nodes = sorted_feature_nodes
        self.nodes_by_feature_id = nodes_by_feature_id

    def _get_col_subset(self, X, feature):
        basic_inputs = list(filter(lambda x: isinstance(x, str), feature.input))
        dependent_inputs = list(filter(lambda x: isinstance(x, FeatureOutput), feature.input))

        Xt = X[basic_inputs]
        for parent_output in dependent_inputs:
            parent_node = self.nodes_by_feature_id[id(parent_output.feature)]
            parent_output_df = parent_node.output
            if parent_output.outputs:
                parent_output_df = parent_output_df[parent_output.outputs]
            Xt = pd.concat([Xt, parent_output_df], axis=1)
        return Xt
    
    def _transform(self, X, y=None, do_fit=False):
        extracted = []
        self.transformed_names_ = []
        for feature_node in self.sorted_feature_nodes:
            fea = feature_node.feature
            # columns could be a string or list of
            # strings; we don't care because pandas
            # will handle either.
            Xt = self._get_col_subset(X, fea)
            if fea.transformer is not None:
                #     if do_fit and hasattr(transformers, 'fit_transform'):
                #         Xt = _call_fit(transformers.fit_transform, Xt, y)
                #     else:
                #         if do_fit:
                #             _call_fit(transformers.fit, Xt, y)
                Xt = fea.transformer.transform(Xt)
                feature_node.output = Xt
            extracted.append(_handle_feature(Xt))

            # alias = options.get('alias')
            # self.transformed_names_ += self.get_names(
            #     columns, transformers, Xt, alias)
        return extracted
    
    def transform(self, X):
        """
        Transform the given data. Assumes that fit has already been called.

        X       the data to transform
        """
        return self._transform(X)

def _handle_feature(fea):
    """
    Convert 1-dimensional arrays to 2-dimensional column vectors.
    """
    if len(fea.shape) == 1:
        fea = np.array([fea]).T

    return fea

def _construct_feature_DAG(features):
    """
    Topologically sort features such that f1 comes before f2 if f2 depends on f1
    @param features the list of features to sort.
    @returns a list of feature nodes, topologically sorted.
    """
    sorted_feature_nodes=[]
    nodes_by_feature_id = {}
    for feature in features:
        if id(feature) not in nodes_by_feature_id:
            _explore_feature_DAG(feature, sorted_feature_nodes, nodes_by_feature_id)
    return (sorted_feature_nodes, nodes_by_feature_id)
    
def _explore_feature_DAG(feature, sorted_feature_nodes, nodes_by_feature_id):
    feature_node = FeatureDAGNode(feature)
    nodes_by_feature_id[id(feature)] = feature_node
    for feature_input in feature.input:
        if isinstance(feature_input, FeatureOutput):
            parent_feature = feature_input.feature
            if id(parent_feature) not in nodes_by_feature_id:
                _explore_feature_DAG(parent_feature, sorted_feature_nodes, nodes_by_feature_id)
            feature_node.add_parent(nodes_by_feature_id[id(parent_feature)])
            nodes_by_feature_id[id(parent_feature)].add_child(feature_node)
    sorted_feature_nodes.append(feature_node)

class FeatureDAGNode:
    def __init__(self, feature, parents=[], children=[]):
        self.feature = feature 
        self.parents = parents
        self.children = children
        self.output = None

    def add_child(self, feature):
        self.children.append(feature)
    
    def add_parent(self, feature):
        self.parents.append(feature)