from {{cookiecutter.package_name}} import problem
from {{cookiecutter.package_name}} import datastream
from {{cookiecutter.package_name}} import architecture

from {{cookiecutter.package_name}}.log_examples import log_examples
from {{cookiecutter.package_name}}.train import train

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution('{{cookiecutter.repository_name}}').version
except DistributionNotFound:
    pass
