from plumed_nodes import __all__


def nodes() -> dict[str, list[str]]:
    """Return all available nodes."""
    return {"plumed-nodes": __all__}
