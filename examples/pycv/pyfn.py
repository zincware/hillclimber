import plumed.plumedCommunications as PLMD
plumedInit={"Value":PLMD.defaults.COMPONENT_NODEV}

def plumedCalculate(action: PLMD.PythonFunction):
    action.lognl("Hello, world!")
    return 0.0