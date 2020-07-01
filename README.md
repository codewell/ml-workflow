# ML Workflow
ML workflow contains our process of bringing a project to fruition as
efficiently as possible. This is subject to change as we iterate and improve.
This package implements tools that add missing features or help bridge the gap
between frameworks and tools that we utilize.

The main packages and tools that we build around are:

- pytorch
- ignite
- pytorch-datastream
- guild

## Install in existing project

    pip install ml-workflow

## MNIST example / Create new project with MNIST template

    mkdir new-project
    cd new-project
    virtualenv venv -p python3.8
    source venv/bin/activate
    pip install ml-workflow
    python -m workflow.setup_project

    pip install -r requirements.txt
    git init

    # reactivate environment to find guild
    deactivate
    source venv/bin/activate

You can train a model and inspect the training with:

    guild run prepare
    guild run train
    guild tensorboard

## Development

### Prepare and run tests

    git clone git@github.com:codewell/ml-workflow.git
    cd ml-workflow
    virtualenv venv --python python3.8
    source venv/bin/activate
    pip install -r requirements.txt
    python -m pytest

### Use development version in project
The following steps will create a link to the local directory and any changes made to the package there will directly carry over to your project environment.

_Packages installed with the editable flag `-e` can behave differently when it comes to imports._

    cd path/to/my/project
    source venv/bin/activate

    cd path/to/work/area
    git clone git@github.com:codewell/ml-workflow.git
    cd ml-workflow
    pip install -e .

## Upload new version to pypi
List current versions

    git fetch --tags
    git tag

Be up-to-date with origin/master with no local changes.

    ./publish.sh <version-without-the-v>
