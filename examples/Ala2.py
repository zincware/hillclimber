import dataclasses

import ipsuite as ips
import zntrack

import hillclimber as hc


@dataclasses.dataclass
class ANIImport:
    model: str = "ANI2x"
    device: str = "cuda"

    def get_calculator(self, **kwargs):
        import torch
        import torchani

        device = torch.device("cuda:0")
        model = torchani.models.ANI2x().to(device)
        return model.ase()


TEMPERATURE = 300.0  # Kelvin
TIMESTEP = 0.5  # fs
N_STEPS = 2_000_000  # Number of MD steps


def main():
    project = zntrack.Project()
    model = ANIImport()

    with project:
        alanine_dipeptide = ips.Smiles2Atoms(smiles="CC(=O)NC(C)C(=O)NC")
        water = ips.Smiles2Conformers(
            smiles="O",
            numConfs=100,
        )

        box = ips.Packmol(
            data=[alanine_dipeptide.frames, water.frames],
            count=[1, 64],
            density=1000,
        )
        geoopt = ips.ASEGeoOpt(
            data=box.frames,
            model=model,
            optimizer="LBFGS",
            run_kwargs={"fmax": 0.05},  # Convergence criterion (eV/Angstrom)
        )

        phi_selector = hc.SMARTSSelector(pattern="[C:1](=O)[N:2][C:3][C:4](=O)")

        psi_selector = hc.SMARTSSelector(pattern="C(=O)[N:1][C:2][C:3](=O)[N:4]")

        psi_cv = hc.TorsionCV(
            atoms=psi_selector,
            prefix="psi",
        )

        phi_cv = hc.TorsionCV(
            atoms=phi_selector,
            prefix="phi",
        )

        phi_bias = hc.MetadBias(
            cv=phi_cv,
            sigma=0.35,
            grid_min="-pi",
            grid_max="pi",
        )

        psi_bias = hc.MetadBias(
            cv=psi_cv,
            sigma=0.35,
            grid_min="-pi",
            grid_max="pi",
        )

        metad_config = hc.MetaDynamicsConfig(
            height=0.1,  # 0.1 eV
            pace=500,
            biasfactor=6,
            temp=TEMPERATURE,
            file="HILLS",
        )

        print_action = hc.PrintAction(cvs=[phi_cv, psi_cv], stride=100, file="COLVAR")

        metad_model = hc.MetaDynamicsModel(
            config=metad_config,
            bias_cvs=[phi_bias, psi_bias],
            actions=[print_action],
            data=geoopt.frames,
            data_idx=-1,
            model=model,
        )

        md_simulation = ips.ASEMD(
            data=geoopt.frames,
            data_ids=-1,
            model=metad_model,
            thermostat=ips.LangevinThermostat(
                temperature=TEMPERATURE, friction=0.01, time_step=TIMESTEP
            ),
            steps=N_STEPS,
            sampling_rate=1000,
            dump_rate=10,
        )

    project.repro()


def eval():
    import metadynminer

    _ = hc.plot_cv_time_series(colvar_file="nodes/ASEMD/model/-1/COLVAR")
    hillsfile = metadynminer.Hills(name="nodes/ASEMD/model/-1/HILLS")
    fes = metadynminer.Fes(hillsfile)
    fes.plot()


if __name__ == "__main__":
    main()
    eval()
