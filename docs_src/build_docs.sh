rm -rf ../docs
sphinx-build -b html ./ ../docs
touch .nojekyll
