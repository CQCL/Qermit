# Building the docs

1. Clone repo checking out submodules

```shell
git clone git@github.com:CQCL/Qermit.git --recurse-submodules
```

2. Install Qermit and docs dependencies

```shell
python -m venv ".venv"
source .venv/bin/activate
pip install -e .[docs] 
```

3. Run `build-docs.sh` script (make sure virtual env is activated)

```shell
cd docs
bash ./build-docs.sh
```

4. Serve html locally (optional)


```shell
npx serve build
```

alternatively, just look at the built html pages in `docs/build`. Serving the html locally allows you to see the navbar.