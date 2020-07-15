from ballet.project import FeatureEngineeringProject

import {{ cookiecutter.package_slug }} as package
from {{ cookiecutter.package_slug }}.features.encoder import get_target_encoder
from {{ cookiecutter.package_slug }}.load_data import load_data


api = FeatureEngineeringProject(
    package=package,
    encoder=get_target_encoder(),
    load_data=load_data,
)
