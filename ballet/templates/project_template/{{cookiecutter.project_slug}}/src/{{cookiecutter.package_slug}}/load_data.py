from ballet.project import load_config
from ballet.util.io import load_table_from_config
from funcy import some, where


def load_data(input_dir=None):
    """Load data"""
    if input_dir is not None:
        config = load_config()
        tables = config.get('data.tables')

        entities_table_name = config.get('data.entities_table_name')
        entities_config = some(where(tables, name=entities_table_name))
        X = load_table_from_config(input_dir, entities_config)

        targets_table_name = config.get('data.targets_table_name')
        targets_config = some(where(tables, name=targets_table_name))
        y = load_table_from_config(input_dir, targets_config)
    else:
        raise NotImplementedError

    return X, y
