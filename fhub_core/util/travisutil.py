import logging
import os
from fhub_core.exc import UnexpectedTravisEnvironmentError
logger = logging.getLogger(__name__)
def get_travis_pr_num():
    '''Return the PR number if the job is a pull request, None otherwise

    See also:
        - <https://docs.travis-ci.com/user/environment-variables
          /#Default-Environment-Variables>
    '''
    try:
        travis_pull_request = get_travis_env_or_fail('TRAVIS_PULL_REQUEST')
        if travis_pull_request == 'false':
            return None
        else:
            try:
                pr_num = int(travis_pull_request)
                return pr_num
            except ValueError:
                return None
    except UnexpectedTravisEnvironmentError:
        return None


def is_travis_pr():
    '''Check if the current job is a pull request build'''
    return get_travis_pr_num() is not None
