# Building the docs

1. Clone repo checking out submodules

```shell
git clone git@github.com:CQCL/Qermit.git --recurse-submodules
```

2. Install Qermit and docs dependencies 

```shell
pip install -e .[docs] 
```

Here an editable wheel is installed with pip. It is also possible to use poetry.

3. Run the `build-docs.sh` script inside `docs`.

```shell
cd docs
bash ./build-docs.sh
```

4. Serve html locally (optional)


```shell
npx serve build
```

Alternatively, just look at the built html pages in `docs/build`. Serving the html locally allows you to see the navbar.