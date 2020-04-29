# Log of workflow

    mkdir my-project
    cd my-project
    git init

    virtualenv venv --python python3.8
    source venv/bin/activate

    pip install ml-workflow
    cookiecutter ??/ml-workflow/template -o ../ -f
    # python -m workflow.setup_project

    pip install -r requirements.txt
