# ML Workflow

## Install

    pip install ml-workflow

## Usage

    from workflow.functional import starcompose

    train_example = starcompose(
        read_example,
        augment,
        preprocess,
    )

## Development
There can be differences with imports when installed with editable `-e` flag.

### Prepare and run tests

    git clone git@github.com:codewell/ml-workflow.git
    cd ml-workflow
    guild init
    source guild-env
    pytest

### Use development version in project
You need to clone and initialize the project as described above before
running this. The following steps will create a link to the local directory
and any changes made to the package there will directly carry over to your
project environment.

    pip uninstall ml-workflow

    cd path/to/my/project
    source guild-env
    cd path/to/ml-workflow
    pip install -e .

## Update package requirements
Modify `install_requires` in `setup.py`.

## Update development requirements

    source guild-env
    pip freeze > requirements.txt

## Upload new version to pypi
Update version in `setup.py` and then:

    source guild-env
    python setup.py sdist
    twine upload dist/ml-workflow-<VERSION>.tar.gz
