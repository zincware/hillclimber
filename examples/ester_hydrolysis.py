import ipsuite as ips

import hillclimber as hc
from src import ANIImport


def main():
    project = ips.Project()
    model = ANIImport()

    thermostat = ips.LangevinThermostat(
        time_step=0.5,
        temperature=400,
        friction=0.01,
    )

    with project:
        methylacetate = ips.Smiles2Conformers(smiles="COC(=O)C", numConfs=50)
        water = ips.Smiles2Conformers(smiles="O", numConfs=50)
        oh = ips.Smiles2Conformers(smiles="[OH-]", numConfs=50)

        box = ips.MultiPackmol(
            data=[methylacetate.frames, oh.frames, water.frames],
            count=[1, 1, 16],
            density=1000,
            n_configurations=1,
        )

        geoopt = ips.ASEGeoOpt(
            data=box.frames,
            model=model,
        )
        s_carbonyl_c = hc.SMARTSSelector(pattern="CO[C:1](=O)C", hydrogens="exclude")

        s_methoxy_o = hc.SMARTSSelector(pattern="C[O:1]C(=O)C", hydrogens="exclude")

        s_all_oxygens = hc.SMARTSSelector(
            pattern="[O;H1,H2;!$(O-C)]", hydrogens="exclude"
        )

        cv_attack_group = hc.CoordinationNumberCV(
            x1=s_all_oxygens,
            x2=s_carbonyl_c,
            prefix="cn_attack_group",
            r_0=2.0,
        )

        cv_leaving = hc.CoordinationNumberCV(
            x1=s_carbonyl_c,
            x2=s_methoxy_o,
            prefix="cn_leaving",
            r_0=2.0,
            d_0=0.0,
        )

        cv1_bias = hc.MetadBias(
            cv=cv_attack_group, sigma=0.05, grid_min=0.0, grid_max=2.0, grid_bin=200
        )
        cv2_bias = hc.MetadBias(
            cv=cv_leaving, sigma=0.1, grid_min=0.0, grid_max=1.0, grid_bin=200
        )

        metad = hc.MetaDynamicsModel(
            config=hc.MetaDynamicsConfig(
                height=0.25, temp=300, pace=2000, biasfactor=10
            ),
            data=geoopt.frames,
            bias_cvs=[cv1_bias, cv2_bias],
            actions=[hc.PrintAction(stride=10, cvs=[cv_attack_group, cv_leaving])],
            model=model,
            timestep=0.5,
        )

        md = ips.ASEMD(
            data=geoopt.frames,
            model=metad,
            thermostat=thermostat,
            steps=1_000_000,
            sampling_rate=100,
            dump_rate=10,
        )

    project.repro()


def eval():
    import metadynminer

    _ = hc.plot_cv_time_series("nodes/ASEMD/model/-1/COLVAR")
    hillsfile = metadynminer.Hills(name="nodes/ASEMD/model/-1/HILLS")
    fes = metadynminer.Fes(hillsfile)
    fes.plot()


if __name__ == "__main__":
    main()
    eval()
