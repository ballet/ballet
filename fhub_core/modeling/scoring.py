import sklearn.metrics
from funcy import flip, partial, rcompose, select_values, suppress

from fhub_core.exc import Error
from fhub_core.modeling.constants import CLASSIFICATION_SCORING, REGRESSION_SCORING, SCORING_NAME_MAPPER
from fhub_core.util.log import logger
from fhub_core.util.modutil import import_module_from_modname


class ScorerInfo:

    def __init__(self, scorer=None, name=None, description=None):
        self._scorer = scorer
        self._name = name
        self._description = description

    @property
    def scorer(self):
        if self._scorer is None:
            with suppress(ValueError):
                self._scorer = sklearn.metrics.get_scorer(self.name)

            # try to import
            try:
                module_name, scorer_name = self.name.rsplit('.', maxsplit=1)
            except ValueError:
                raise ValueError(
                    'Invalid scorer import path: {}'.format(self.name))
            try:
                mod = import_module_from_modname(module_name)
                self._scorer = getattr(mod, scorer_name)
            except (ImportError, ModuleNotFoundError):
                logger.exception(
                    'Failed to import scorer variable {scorer_name} from module {module_name}'
                    .format(scorer_name=scorer_name, module_name=module_name))

        if self._scorer is not None:
            return self._scorer
        else:
            raise Error(
                'Could not get a scorer with configuration {}'.format(self.name))

    @property
    def name(self):
        if self._name is None:
            if self._scorer is not None:
                # try from scorer
                if isinstance(self._scorer, sklearn.metrics.scorer._BaseScorer):
                    scorers = sklearn.metrics.scorer.SCORERS
                    matches = select_values(lambda x: x == self._scorer, scorers)
                    matches = list(matches.keys())
                    if len(matches) == 1:
                        self._name = matches[0]
                    elif len(matches) > 1:
                        # unexpected
                        logger.debug(
                            'Unexpectedly found multiple matches for scorer name {name}: '
                            '{matches!r}'
                            .format(name=self._name, matches=matches))
                    else:
                        # must be a custom scorer, try to get name
                        if hasattr(self._scorer, '__name__'):
                            self._name = self._scorer.__name__
            elif self._description is not None:
                # try from description
                mapper = flip(SCORING_NAME_MAPPER)
                if self._description in mapper:
                    self._name = mapper[self._description]
                else:
                    # default formatting
                    self._name = '_'.join(self._description.lower().split(' '))

        if self._name is not None:
            return self._name
        else:
            raise Error('Could not get name from scorer')

    @property
    def description(self):
        if self._description is None:
            name = self.name

            if name in SCORING_NAME_MAPPER:
                self._description = SCORING_NAME_MAPPER[name]
            else:
                # default formatting
                def upper_first(s):
                    return s[0].upper() + s[1:] if s is not None else s
                format = rcompose(
                    lambda s: s.split('_'),
                    partial(map, upper_first),
                    lambda l: ' '.join(l),
                )
                self._description = format(name)

        return self._description


def get_scorer_names_for_problem_type(problem_type):
    '''Get scorers for this problem type.

    Returns:
        list: List of scorer_name as defined in sklearn.metrics. This is a 'utility
            variable' that can be used where we just need the names of the
            scoring functions and not the more complete information.
    '''
    # scoring_types maps user-readable name to `scoring`, as argument to
    # cross_val_score
    # See also
    # http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    if problem_type.is_classification():
        return CLASSIFICATION_SCORING
    elif problem_type.is_regression():
        return REGRESSION_SCORING
    else:
        raise NotImplementedError
