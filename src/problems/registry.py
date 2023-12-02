from problems.Py2048 import Py2048

def getProblem(name):
    if name == 'Py2048':
        return Py2048

    raise NotImplementedError()
