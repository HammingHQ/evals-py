# https://packaging.python.org/en/latest/tutorials/packaging-projects/
# python3 -m pip install --upgrade twine

rm -rf dist
python -m build
python3 -m twine upload dist/*
