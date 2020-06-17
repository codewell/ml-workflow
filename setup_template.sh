#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail -o xtrace

VERSION=${1:-"3.8"}

rm -rf test_template
mkdir test_template
cd test_template
virtualenv venv -p "python${VERSION}"
source venv/bin/activate
cd ..
pip install -e .
cd test_template
echo "test
test
0.1.0
test project
y" | python -m workflow.setup_project
pip install -r requirements.txt
deactivate
