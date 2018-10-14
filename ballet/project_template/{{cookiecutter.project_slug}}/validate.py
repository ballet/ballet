#!/usr/bin/env python

if __name__ == '__main__':
    import logging

    import ballet.util.log
    import ballet.validation

    import {{ cookiecutter.project_slug }}

    ballet.util.log.enable(level=logging.DEBUG, echo=False)

    ballet.validation.main({{ cookiecutter.project_slug }})
