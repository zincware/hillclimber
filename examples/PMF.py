import dataclasses
from pathlib import Path

import ipsuite as ips
import zntrack

import hillclimber as hc

# Configure the simulation parameters
TEMPERATURE = 300.0  # Kelvin
TIMESTEP = 0.5  # fs
N_STEPS = 5_000_000  # Number of MD steps


alanine_smiles = "C[C@H](N)C(=O)O"
cysteine_smiles = "N[C@@H](CS)C(=O)O"


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


class FASTAtoms(zntrack.Node):
    """Convert FASTA sequence to Atoms object using RDKit."""

    text: str = zntrack.params()
    frames_path: Path = zntrack.outs_path(zntrack.nwd / "frames.xyz")
    smiles: str = zntrack.outs()

    def run(self):
        import ase.io
        import molify
        from rdkit import Chem

        mol = Chem.MolFromFASTA(self.text)
        self.smiles = Chem.MolToSmiles(mol)
        frames = molify.rdkit2ase(mol)
        ase.io.write(self.frames_path, frames)

    @property
    def frames(self) -> list:
        import ase.io

        return list(ase.io.iread(self.frames_path))


def main():
    project = zntrack.Project()

    model = ANIImport()

    with project:
        water = ips.Smiles2Atoms(smiles="O")  # Water molecule
        alanine = FASTAtoms(text="A")  # Alanine residue
        cysteine = FASTAtoms(text="C")  # Cysteine residue

        small_box = ips.Packmol(
            data=[water.frames, alanine.frames, cysteine.frames],
            count=[16, 1, 1],
            density=1000,
            tolerance=2.5,
        )
        data_gen = ips.Packmol(
            data=[small_box.frames, water.frames],
            count=[1, 96],
            density=1000,
            tolerance=2.5,
        )
        relax = ips.ASEGeoOpt(
            data=data_gen.frames, model=model, run_kwargs={"fmax": 0.1}
        )

        a_selector = hc.SMILESSelector(smiles=alanine_smiles)
        c_selector = hc.SMILESSelector(smiles=cysteine_smiles)

        distance_cv = hc.DistanceCV(
            x1=hc.VirtualAtom(atoms=a_selector, reduction="com"),
            x2=hc.VirtualAtom(atoms=c_selector, reduction="com"),
            prefix="d_a_c",
        )

        # SIGMA about ~1/5 to 1/10 of the CV fluctuation range
        metad_bias = hc.MetadBias(
            cv=distance_cv, sigma=0.75, grid_min=0.0, grid_max=8.0, grid_bin=300
        )
        # Upper Wall
        upper_wall = hc.UpperWallBias(
            cv=distance_cv,
            at=7.5,
            kappa=5.0,  # Force constant (eV/Angstrom²)
        )
        metad_config = hc.MetaDynamicsConfig(
            height=0.03,  # eV
            pace=500,
            biasfactor=10.0,
            temp=TEMPERATURE,
            file="HILLS",
        )
        print_action = hc.PrintAction(
            cvs=[distance_cv],
            stride=10,
            file="COLVAR",
        )

        metad_model = hc.MetaDynamicsModel(
            config=metad_config,
            bias_cvs=[metad_bias],
            actions=[print_action, upper_wall],
            data=relax.frames,
            data_idx=-1,
            model=model,
            timestep=TIMESTEP,
        )

        ips.ASEMD(
            data=relax.frames,
            data_ids=-1,
            model=metad_model,
            thermostat=ips.LangevinThermostat(
                temperature=TEMPERATURE, friction=0.01, time_step=TIMESTEP
            ),
            steps=N_STEPS,
            sampling_rate=100,
            dump_rate=10,
        )

    project.repro()


def eval():
    import matplotlib.pyplot as plt
    import numpy as np

    _ = hc.plot_cv_time_series("nodes/ASEMD/model/-1/COLVAR")
    hc.sum_hills(
        hills_file="nodes/ASEMD/model/-1/HILLS",
        bin=500,
        outfile="fes.dat",
    )

    # Load the fes.dat file
    data = np.loadtxt("fes.dat", comments="#")

    # Extract columns
    d_na_cl = data[:, 0]  # Collective variable (distance)
    free_energy = data[:, 1]  # Free energy

    # Create the plot
    _, (ax1) = plt.subplots(1, 1, sharex=True)

    # Plot free energy surface
    ax1.plot(d_na_cl, free_energy, "b-", linewidth=2)
    ax1.set_ylabel("Free Energy (kJ/mol)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=7.5, color="k", linestyle="--", alpha=0.3)
    ax1.axvline(x=15.5 / 2, color="k", linestyle="-.", alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
