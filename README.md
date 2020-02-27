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

### Prepare and run tests

    git clone git@github.com:codewell/ml-workflow.git
    cd ml-workflow
    guild init
    source guild-env
    python -m pytest

### Use development version in project
The following steps will create a link to the local directory and any changes made to the package there will directly carry over to your project environment.

_Packages installed with the editable flag `-e` can behave differently when it comes to imports._

    cd path/to/my/project
    source guild-env
    pip uninstall ml-workflow

    cd path/to/work/area
    git clone git@github.com:codewell/ml-workflow.git
    cd ml-workflow
    pip install -e .

## Update package requirements
Edit requirements.txt

## Upload new version to pypi
List current versions

    git fetch --tags
    git tag

Be up-to-date with origin/master with no local changes.

    ./publish.sh <version-without-the-v>
