import ballet.client
import requests

from funcy import decorator
from notebook.notebookapp import list_running_servers
servers = list(list_running_servers())
server = servers[0] if len(servers) == 1 else None


def event(name, details):
    if server is not None:
        requests.post(
            server['url'] + 'ballet/tel',
            headers={'token': server['token']},
            json={'name': name, 'details': details})


@decorator
def instrument(call):
    event_name = call._func.__qualname__
    result = call()
    event(event_name, {'result': result})
    return result


def install():
    ballet.client.Client.validate_feature_api = instrument(
        ballet.client.Client.validate_feature_api)
    ballet.client.Client.validate_feature_acceptance = instrument(
        ballet.client.Client.validate_feature_acceptance)
    ballet.client.Client.api = instrument(
        ballet.client.Client.api)
