from setuptools import setup

version = {}
with open("_version.py") as fp:
    exec(fp.read(), version)

setup(
    name="qermit",
    version=version["__version__"],
    python_requires=">=3.6",
    description="error-mitigation framework, an extension to pytket",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="CQC Non-Commercial Use Software Licence",
    include_package_data=True,
    install_requires=["pytket ~= 0.11.0", "matplotlib", "networkx"],
    classifiers=[
        "Environment :: Console",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: Other/Proprietary License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    zip_safe=False,
)
