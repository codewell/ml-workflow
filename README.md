# ML Workflow

## Install

    pip install git+https://github.com/codewell/ml-workflow.git

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
