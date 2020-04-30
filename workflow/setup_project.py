from cookiecutter.main import cookiecutter
from cookiecutter.exceptions import RepositoryNotFound


if __name__ == '__main__':
    try:
        cookiecutter('/'.join(__file__.split('/')[:-5]) + '/workflow/template')
    except RepositoryNotFound:
        # Probably installed with the -e flag
        cookiecutter('/'.join(__file__.split('/')[:-2]) + '/template')
