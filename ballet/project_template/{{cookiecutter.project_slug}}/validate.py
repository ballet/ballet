#!/usr/bin/env python

if __name__ == '__main__':

    import ballet.util.log
    import ballet.validation

    import {{ cookiecutter.project_slug }}


    ballet.util.log.enable(level='DEBUG', echo=False)
    ballet.validation.main({{ cookiecutter.project_slug }})
