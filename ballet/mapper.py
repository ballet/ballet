import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper

from ballet.feature import Feature, FeatureOutput

class DependenceDataFrameMapper(DataFrameMapper):
    def __init__(self, features):
        if not isinstance(features, list):
            features = [features]
        self.features = features
        (sorted_feature_nodes, feature_nodes_by_src) = _construct_feature_DAG(features)
        self.sorted_feature_nodes = sorted_feature_nodes
        self.feature_nodes_by_src = feature_nodes_by_src

    def _get_col_subset(self, X, feature):
        Xt = X[feature.input.filter(lambda x: isinstance(x, str))]
        dependent_inputs = feature.input.filter(lambda x: isinstance(X, FeatureOutput))
        for parent_output in dependent_inputs:
            parent_node = self.feature_nodes_by_src[parent_output.feature.source]
            parent_output_df = parent_node.output
            if parent_output.input:
                parent_output_df = parent_output_df[parent_output.input]
            Xt = pd.concat([Xt, parent_output_df], axis=1)
        return Xt
    
    def _transform(self, X, y=None, do_fit=False):
        extracted = []
        self.transformed_names_ = []
        for feature_node in self.sorted_feature_nodes:

            # columns could be a string or list of
            # strings; we don't care because pandas
            # will handle either.
            Xt = self._get_col_subset(X, columns)
            if transformers is not None:
                with add_column_names_to_exception(columns):
                #     if do_fit and hasattr(transformers, 'fit_transform'):
                #         Xt = _call_fit(transformers.fit_transform, Xt, y)
                #     else:
                #         if do_fit:
                #             _call_fit(transformers.fit, Xt, y)
                        Xt = transformers.transform(Xt)
            extracted.append(_handle_feature(Xt))

            alias = options.get('alias')
            self.transformed_names_ += self.get_names(
                columns, transformers, Xt, alias)

def add_column_names_to_exception(column_names):
    # Taken from sklearn_pandas.dataframe_mapper, which itself is
    # Stolen from https://stackoverflow.com/a/17677938/356729
    try:
        yield
    except Exception as ex:
        if ex.args:
            msg = u'{}: {}'.format(column_names, ex.args[0])
        else:
            msg = str(column_names)
        ex.args = (msg,) + ex.args[1:]
        raise

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
    feature_nodes_by_src = {}
    feature_nodes = []
    for feature in features:
        if feature.source not in feature_nodes_by_src:
            _explore_feature_DAG(feature, sorted_feature_nodes, feature_nodes_by_src)
    return (sorted_feature_nodes, feature_nodes_by_src)
    
def _explore_feature_DAG(feature, sorted_feature_nodes, feature_nodes_by_src):
    feature_node = FeatureDAGNode(feature)
    feature_nodes_by_src[feature.source] = feature_node
    for feature_input in feautre.input:
        if isinstance(feature_input, FeatureOutput):
            parent_feature = feature_input.feature
            if parent_feature.source not in feature_nodes_by_src:
                _explore_feature_DAG(parent_feature, sorted_feature_nodes, feature_nodes_by_src)
            feature_node.add_parent(feature_nodes_by_src[parent_feature.source])
            feature_nodes_by_src[parent_feature.source].add_child(feature_node)
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