PYTHONPATH=. py.test
PYTHONPATH=. python test/test_doctests.py
PYTHONPATH=. mypy --html-report mypy_report --ignore-missing-imports --follow-imports=skip pyqae
flake8 . --count --select=E901,E999,F821,F822,F823 --statistics
