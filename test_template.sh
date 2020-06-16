#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail -o xtrace

VERSION=${1:-"3.8"}

rm -rf test_template
mkdir test_template
cd test_template
virtualenv venv -p "python${VERSION}"
source venv/bin/activate
cd ..
pip install .
cd test_template
echo "test
test
0.1.0
test project
y" | python -m workflow.setup_project
pip install -r requirements.txt
guild run prepare -y
guild run search_lr n_batches=10 -y
guild run train max_epochs=2 n_batches_per_epoch=2 -y
guild run retrain max_epochs=1 n_batches_per_epoch=1 -y
guild run evaluate -y
deactivate
