#!/bin/sh
sed "s/REQUIREMENTS/$(sed -e 's/[\&/]/\\&/g' -e 's/$/\\n/' ../requirements.txt | tr -d '\n')/" index-rst-template > index.rst
rm -rf ../docs/manual
sphinx-build -b html ./ ../docs/manual -W
rm index.rst
