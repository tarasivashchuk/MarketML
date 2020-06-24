#!/usr/bin/env bash
source /Users/taras/Code/virtual-environments/marketml/bin/activate
/Users/taras/.poetry/bin/poetry update
/Users/taras/.poetry/bin/poetry install
/Users/taras/.poetry/bin/poetry export -f "requirements.txt" -o "requirements.txt" --without-hashes
