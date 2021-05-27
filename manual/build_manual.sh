#!/bin/sh
sed "s/REQUIREMENTS/$(sed -e 's/[\&/]/\\&/g' -e 's/$/\\n/' ../requirements.txt | tr -d '\n')/" index-rst-template > index.rst
rm -rf ./_build
sphinx-build -b html ./ ./_build

rm index.rst
