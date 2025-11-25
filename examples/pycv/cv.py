import jax
import jax.numpy as jnp
import plumedCommunications as PLMD

plumedInit = {"Value": PLMD.defaults.COMPONENT}


@jax.jit
def distance(x):
    diff = x[1] - x[0]
    return jnp.sqrt(jnp.sum(diff**2))


grad_distance = jax.jit(jax.grad(distance))


def cv1(action: PLMD.PythonCVInterface):
    x = action.getPositions()

    val = distance(x)
    grads = grad_distance(x)

    box_grads = jnp.zeros((3, 3))

    return float(val), grads, box_grads
