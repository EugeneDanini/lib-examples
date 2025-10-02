#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Error: library name"
    exit 1
fi

pip freeze | xargs pip uninstall -y
pip install -r requirements.txt
pip install "$1"
pip freeze > requirements.txt

directory=$(echo "$1" | tr '-' '_')
directory="lib_${directory}"
mkdir "$directory" && touch "$directory/__init__.py" && touch "$directory/example.py"
