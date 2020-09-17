# {{cookiecutter.package_name}}

## Installation
```
virtualenv venv --python python3.8
source venv/bin/activate
pip install -r requirements.txt
guild run prepare
```

## Training
```
guild run prepare
guild run train
guild run retrain model=<model-hash>
guild run evaluate model=<model-hash>
```
