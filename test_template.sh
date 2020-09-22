#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail -o xtrace

cd test_template
shopt -s extglob 
rm -rf .github
rm -rf !(venv)
source venv/bin/activate
echo "repository
package
0.1.0
project description
y" | python -m workflow.setup_project
guild run prepare -y
guild run search_lr n_batches=10 -y
guild run train max_epochs=2 n_batches_per_epoch=2 -y
guild run retrain max_epochs=1 n_batches_per_epoch=1 -y
guild run evaluate -y
deactivate
