echo "clear the dist directory ... "
rm -f dist/*

echo "generate the whl file ... "
python setup.py sdist bdist_wheel

echo "install new version ... "
pip uninstall -y llmplus
pip install dist/*.whl

echo "show information ... "
pip show llmplus
