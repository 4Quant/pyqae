"""
mypy pyqae

sphinx-apidoc -F -o docs/ pyqae/
"""
import os
from subprocess import Popen, PIPE
exec_cmd = lambda prog, pname: Popen([prog, pname], shell=True, stdout=PIPE).stdout.read()
if __name__ == "__main__":
    from mypy.main import main

    main(os.path.join('..','pyqae'))

    pydoc_path = exec_cmd("/usr/bin/which", "pydoc")
    exec_cmd(pydoc_path, "pyqae")
