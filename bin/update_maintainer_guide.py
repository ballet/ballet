#!/usr/bin/env python3

import json
import pathlib
import tempfile
from invoke import Context, Responder

c = Context()

# TODO: has error Bad file descriptor
# class MyRunner(Local):
#     def _write_proc_stdin(self, data):
#         super()._write_proc_stdin(data)
#         # now also write to stdout
#         os.write(self.process.stdout.fileno(), data)
#
# runner = MyRunner(c)

ROOT = pathlib.Path(__file__).resolve().parent.parent
COOKIECUTTER_PATH = ROOT.joinpath(
    'ballet', 'templates', 'project_template', 'cookiecutter.json')

TMP = tempfile.mkdtemp(dir='/tmp')

# TODO
# import atexit
# atexit.register(_d.cleanup)

snippets = []

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

# ballet quickstart snippet
with c.cd(TMP):
    r = c.run('ballet quickstart',
              watchers=[Responder(t[0], t[1] + '\n') for t in responders],
              hide=True)
    snippets.append(r.stdout)

# tree snippet
with c.cd(TMP):
    r = c.run('tree -a ballet-my-project -I .git', hide=True)
    snippets.append(r.stdout)

# got these snippets
for snippet in snippets:
    print(snippet)
    print()

print(f'working in {TMP}')
