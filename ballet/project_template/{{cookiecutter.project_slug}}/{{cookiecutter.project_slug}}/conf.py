from ballet.compat import pathlib
from ballet.project import make_config_get


def here():
    return pathlib.Path(__file__).resolve()


get = make_config_get(here())
