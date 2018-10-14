#!/usr/bin/env bash

git init
git config --local --add user.name "{{ cookiecutter.full_name }}"
git config --local --add user.email "{{ cookiecutter.email }}"
git config --local --add github.user "{{ cookiecutter.github_username }}"
git add .
git commit -m "Automatically generated files from ballet-quickstart"
git remote add origin "https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}"
