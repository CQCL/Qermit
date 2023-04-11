rm -rf ../docs
sphinx-build -b html ./ ../docs -W
touch ../docs/.nojekyll
