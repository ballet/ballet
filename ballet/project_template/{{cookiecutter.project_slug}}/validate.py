#!/usr/bin/env python

if __name__ == '__main__':
    import {{ cookiecutter.project_slug }}
    import ballet.validation

    ballet.validation.main({{ cookiecutter.project_slug }})
