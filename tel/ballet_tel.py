import requests
from funcy import contextmanager, decorator, silent
from notebook.notebookapp import list_running_servers

servers = list(list_running_servers())
server = servers[0] if len(servers) == 1 else None

import ballet.client  # noqa

call_depth = 0


@silent
def event(name, details):
    if call_depth == 0:
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
def instrument(call):
    event_name = call._func.__qualname__

    try:
        with disable_events():
            result = call()
        event(event_name, {'result': result, 'error': False})
        return result
    except Exception:
        event(event_name, {'result': None, 'error': True})
        raise


def instrumentproperty(oldprop):

    @property
    def prop(self):
        event_name = oldprop.fget.__qualname__
        event(event_name, {})
        with disable_events():
            return oldprop.fget(self)

    return prop


def install():
    ballet.client.Client.validate_feature_api = instrument(
        ballet.client.Client.validate_feature_api)
    ballet.client.Client.validate_feature_acceptance = instrument(
        ballet.client.Client.validate_feature_acceptance)
    ballet.client.Client.api = instrumentproperty(
        ballet.client.Client.api)
