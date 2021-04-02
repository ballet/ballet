#!/usr/bin/env python3

import atexit
import json
import pathlib
import tempfile
from invoke import Context, Responder
from invoke.config import Config
from invoke.runners import Local
from stacklog import stacktime


class MyRunner(Local):
    def _write_proc_stdin(self, data):
        super()._write_proc_stdin(data)

        # now also write to stdout
        # this is hacky... we know that Local.handle_stdout takes strs and
        # appends them to a buffer that is shared between io threads. at the
        # end of execution, the contents of the buffer are joined into a
        # string. so we can "write" to the "stdout" of the underlying process
        # by just appending data to this thread.
        self.stdout.append(self.decode(data))


overrides = {'runners': {'local': MyRunner}}
config = Config(overrides=overrides)
c = Context(config=config)

_d = tempfile.TemporaryDirectory(dir='/tmp')
atexit.register(_d.cleanup)

ROOT = pathlib.Path(__file__).resolve().parent.parent
COOKIECUTTER_PATH = ROOT.joinpath(
    'ballet', 'templates', 'project_template', 'cookiecutter.json')
FRAGMENTS_DIR = ROOT.joinpath(
    'docs', 'fragments', 'maintainer-guide')
TMP = pathlib.Path(_d.name)

responders = {
    'full_name': 'Jane Developer',
    'email': 'jane@developer.org',
    'github_owner': 'jane_developer',
    'project_name': 'Predict my thing',
    'project_slug': 'ballet-my-project',
    'package_slug': 'myproject',
    'problem_type': '2',
    'classification_type': '1',
    'classification_scorer': '1',
    'regression_scorer': '5',
    'pruning_action': '3',
    'auto_merge_accepted_features': '2',
    'auto_close_rejected_features': '2',
}

# check keys
with COOKIECUTTER_PATH.open('r') as f:
    j = json.load(f)
prompts = [key for key in j.keys() if not key.startswith('_')]
difference = set(responders.keys()) ^ set(prompts)
assert not difference, f'had these element in only one keyset: {difference}'


def create_quickstart_fragment(dir, path):
    cmd = 'ballet quickstart'
    with c.cd(dir):
        r = c.run(
            cmd,
            watchers=[
                Responder(prompt, response + '\n')
                for prompt, response
                in responders.items()
            ],
            echo=True,
            hide=True,
        )

    with open(path, 'w') as f:
        f.write(f'$ {cmd}\n')
        f.write(r.stdout)


def create_tree_fragment(dir, path):
    cmd = 'tree -a ballet-my-project -I ".git|__pycache__"'
    with c.cd(dir):
        r = c.run(cmd, echo=True, hide=True)

    with open(path, 'w') as f:
        f.write(f'$ {cmd}\n')
        f.write(r.stdout)


def create_gitlog_fragment(dir, path):
    content = []
    for cmd in [
        'git log',
        'git remote -v',
    ]:
        with c.cd(dir):
            r = c.run(cmd, echo=True, hide=True)
        content.append(f'$ {cmd}')
        content.append(r.stdout)
        content.append('')

    with open(path, 'w') as f:
        f.write('\n'.join(content))


print(f'working in {TMP}')


with stacktime(print, 'creating ballet quickstart fragment'):
    create_quickstart_fragment(
        TMP, FRAGMENTS_DIR.joinpath('ballet-quickstart.txt'))
with stacktime(print, 'creating tree project fragment'):
    create_tree_fragment(
        TMP, FRAGMENTS_DIR.joinpath('tree-project.txt'))
with stacktime(print, 'creating git log fragment'):
    create_gitlog_fragment(
        TMP.joinpath('ballet-my-project'),
        FRAGMENTS_DIR.joinpath('git-log.txt'))
