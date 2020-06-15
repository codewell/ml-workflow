#!/usr/bin/env bash

rm -rf test_template
mkdir test_template
cd test_template
virtualenv venv -p python3.8
source venv/bin/activate
cd ..
pip install .
cd test_template
echo "test
test
0.1.0
test project
TestNet
y" | python -m workflow.setup_project
pip install -r requirements.txt
guild run prepare -y
guild run train max_epochs=2 n_batches_per_epoch=2 -y
deactivate
