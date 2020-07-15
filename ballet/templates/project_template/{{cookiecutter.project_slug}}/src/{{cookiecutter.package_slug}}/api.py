from typing import Callable, List, Tuple

from pandas import DataFrame

from ballet.contrib import collect_contrib_features
from ballet.feature import Feature
from ballet.pipeline import make_build, FeatureEngineeringPipeline, BuildResult
from ballet.eng import BaseTransformer

import {{ cookiecutter.package_slug }} as pkg
from {{ cookiecutter.package_slug }}.features.encoder import get_target_encoder
from {{ cookiecutter.package_slug }}.load_data import load_data as _load_data

# --- begin public api ---

features: List[Feature] = collect_contrib_features(pkg)

pipeline: FeatureEngineeringPipeline = FeatureEngineeringPipeline(features)

encoder: BaseTransformer = get_target_encoder()

load_data: Callable[..., Tuple[DataFrame, DataFrame]] = _load_data

build: Callable[..., BuildResult] = make_build(pipeline, encoder, load_data)

# --- end public api ---
