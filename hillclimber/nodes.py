from hillclimber import __all__


def nodes() -> dict[str, list[str]]:
    """Return all available nodes."""
    return {"hillclimber": __all__}
