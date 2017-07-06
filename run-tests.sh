PYTHONPATH=. py.test
PYTHONPATH=. python test/test_doctests.py
PYTHONPATH=. mypy --html-report mypy_report --ignore-missing-imports --follow-imports=skip pyqae
