import sklearn.metrics
from funcy import flip, partial, rcompose

from fhub_core.exc import Error
from fhub_core.modeling.constants import  SCORING_NAME_MAPPER
from fhub_core.util.modutil import import_module_from_modname


def get_scorer(scorer_name):
    try:
        scoring = sklearn.metrics.get_scorer(scorer_name)
        found = True
    except ValueError:
        found = False

    if not found:
        i = scorer_name.rfind('.')
        if i < 0:
            raise ValueError(
                'Invalid scorer import path: {}'.format(scorer_name))
        module_name, scorer_name_ = scorer_name[:i], scorer_name[i + 1:]
        mod = import_module_from_modname(module_name)
        scoring = getattr(mod, scorer_name_)
        found = True

    if not found:
        raise Error(
            'Could not get a scorer with configuration {}'.format(scorer_name))

    return scoring


def scoring_name_to_name(scoring_name):
    mapper = SCORING_NAME_MAPPER

    if scoring_name in mapper:
        return mapper[scoring_name]
    else:
        # default formatting
        def upper_first(s):
            if not s:
                return s
            elif len(s) == 1:
                return s.upper()
            else:
                return s[0].upper() + s[1:]
        format = rcompose(
            lambda s: s.split('_'),
            partial(map, upper_first),
            lambda l: ' '.join(l),
        )
        return format(scoring_name)


def name_to_scoring_name(name):
    mapper = flip(SCORING_NAME_MAPPER)
    if name in mapper:
        return mapper[name]
    else:
        # default formatting
        return '_'.join(name.lower().split(' '))
