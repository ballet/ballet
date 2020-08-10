import logging
import requests
from funcy import contextmanager, decorator, silent
from notebook.notebookapp import list_running_servers

servers = list(list_running_servers())
server = servers[0] if len(servers) == 1 else None

call_depth = 0
logger = logging.getLogger('ballettel')


@silent
def event(name, details):
    if call_depth == 0:
        logger.info(f'Got event {name} with details {details}')
        if server is not None:
            requests.post(
                server['url'] + 'ballet/tel',
                headers={'token': server['token']},
                json={'name': name, 'details': details})


@contextmanager
def disable_events():
    global call_depth
    call_depth += 1
    try:
        yield
    finally:
        call_depth -= 1


@decorator
def _instrument(call, post_result):
    event_name = call._func.__qualname__

    try:
        with disable_events():
            result = call()
        details = {'result': result if post_result else None, 'error': False}
        event(event_name, details)
        return result
    except Exception:
        event(event_name, {'result': None, 'error': True})
        raise


instrument = _instrument(False)
instrument_with_result = _instrument(True)


def instrumentproperty(oldprop):

    @property
    def prop(self):
        event_name = oldprop.fget.__qualname__
        event(event_name, {})
        with disable_events():
            return oldprop.fget(self)

    return prop


def install():
    import ballet.client  # noqa
    import ballet.project  # noqa

    ballet.client.Client.validate_feature_api = instrument_with_result(
        ballet.client.Client.validate_feature_api)
    ballet.client.Client.validate_feature_acceptance = instrument_with_result(
        ballet.client.Client.validate_feature_acceptance)
    #ballet.client.Client.api = instrumentproperty(ballet.client.Client.api)
    ballet.project.FeatureEngineeringProject.load_data = instrument(
        ballet.project.FeatureEngineeringProject.load_data)
    ballet.project.FeatureEngineeringProject.engineer_features = instrument(
        ballet.project.FeatureEngineeringProject.engineer_features)
    ballet.project.FeatureEngineeringProject.features = instrumentproperty(
        ballet.project.FeatureEngineeringProject.features)
    ballet.project.FeatureEngineeringProject.pipeline = instrumentproperty(
        ballet.project.FeatureEngineeringProject.pipeline)
