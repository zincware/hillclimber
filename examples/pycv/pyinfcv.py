import sys
import os

# FORCE GLOBAL SYMBOLS
# This ensures that C++ extensions (like PLUMED and PyCV) can share 
# runtime type information and memory structures.
try:
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
except AttributeError:
    pass # Not a posix system, but since you are on mac, this will run.

# Ensure current directory is in path so PLUMED can find 'pyinfcv'
sys.path.insert(0, os.getcwd())

with open("pyinfcv_import.log", "a") as f:
    f.write("pyinfcv module imported\n")

import jax
import jax.numpy as jnp
import plumed.plumedCommunications as PLMD

# 1. Enable double precision (Crucial for stability)
jax.config.update("jax_enable_x64", True)

# 2. Configure defaults
plumedInit = {"Value": PLMD.defaults.COMPONENT}

# 3. Define pure JAX math
@jax.jit
def distance(x):
    # x shape is (2, 3)
    diff = x[1] - x[0]
    return jnp.sqrt(jnp.sum(diff**2))

# 4. Auto-differentiate
grad_distance = jax.jit(jax.grad(distance))

# 5. The PLUMED Interface Function
def cv1(action: PLMD.PythonCVInterface):
    # Get positions (N, 3)
    x = action.getPositions()
    
    # Calculate Value & Gradients
    val = distance(x)
    grads = grad_distance(x)
    
    # --- THE FIX IS HERE ---
    # You MUST return a 3rd element for the virial (box derivatives).
    # Since we aren't changing the box, we return a 3x3 zero matrix.
    box_grads = jnp.zeros((3, 3))
    
    # Return strict tuple: (float, array, array)
    return float(val), grads, box_grads