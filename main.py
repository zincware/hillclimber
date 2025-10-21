import ase

import hillclimber as pn

s1 = pn.IndexSelector(indices=[[0]])
s2 = pn.IndexSelector(indices=[[1]])
distance_cv = pn.DistanceCV(x1=s1, x2=s2, prefix="d12", group_reduction="com")
print_action = pn.PrintActionNode(cv=distance_cv, stride=10, file="output.txt")

bias = pn.MetaDBiasCV(cv=distance_cv, sigma=0.1, grid_min=0.0, grid_max=10.0)

meta_d = pn.MetaDynamicsConfig(height=0.5, pace=150)

meta_d_model = pn.MetaDynamicsModel(
    config=meta_d, bias_cvs=[bias], data=ase.Atoms(), actions=[print_action]
)


import hillclimber as pn

s1 = pn.IndexSelector(indices=[[0]])
s2 = pn.IndexSelector(indices=[[1]])
distance_cv = pn.DistanceCV(x1=s1, x2=s2, prefix="d12", group_reduction="com")
bias = pn.MetaDBiasCV(cv=distance_cv, sigma=0.1, grid_min=0.0, grid_max=10.0)
meta_d = pn.MetaDynamicsConfig(height=0.5, pace=150)
meta_d_model = pn.MetaDynamicsModel(
    config=meta_d, bias_cvs=[bias], data=geom_opt.frames, actions=[print_action]
)

md = ips.ASEMD(
    data=geom_opt.frames,
    model=meta_d_model,
    thermostat=thermostat,
    steps=10_000,
)