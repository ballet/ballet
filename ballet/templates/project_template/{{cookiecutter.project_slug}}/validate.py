#!/usr/bin/env python

if __name__ == '__main__':

    from ballet.util.log import enable
    from ballet.validation.main import validate

    import {{ cookiecutter.project_slug }}

    enable(level='DEBUG', echo=False)
    validate({{ cookiecutter.project_slug }})
