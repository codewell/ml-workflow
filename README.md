# ML Workflow

## Install in existing project

    pip install ml-workflow

## Create new project with template
1. Clone and install ml-workflow

        git clone git@github.com:codewell/ml-workflow.git
        cd ml-workflow
        virtualenv venv --python python3.8
        source venv/bin/activate
        pip install -r requirements.txt

2. Create project with cookiecutter

        cd ..
        cookiecutter ml-workflow/template
        deactivate

3. Activate and install environment

        cd new-project
        virtualenv venv --python python3.8
        source venv/bin/activate
        pip install -r requirements.txt
        git init

## Development

### Prepare and run tests

    git clone git@github.com:codewell/ml-workflow.git
    cd ml-workflow
    virtualenv venv --python python3.8
    source venv/bin/activate
    python -m pytest

### Use development version in project
The following steps will create a link to the local directory and any changes made to the package there will directly carry over to your project environment.

_Packages installed with the editable flag `-e` can behave differently when it comes to imports._

    cd path/to/my/project
    source venv/bin/activate
    pip uninstall ml-workflow

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
