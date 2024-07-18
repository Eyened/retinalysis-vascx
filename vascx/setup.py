from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="vascx",
    # using versioneer for versioning using git tags
    # https://github.com/python-versioneer/python-versioneer/blob/master/INSTALL.md
    # version=versioneer.get_version(),
    # cmdclass=versioneer.get_cmdclass(),
    author="Jose Vargas",
    author_email="j.vargasquiros@erasmusmc.nl",
    description="Retinal analysis toolbox for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    entry_points={"console_scripts": []},
    install_requires=[
        "sknw==0.14",
    ],
    python_requires=">=3.9",
)
