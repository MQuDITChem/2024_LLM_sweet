import argparse
import contextlib
import os
import random
import string
import subprocess

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

rdBase.DisableLog("rdApp.error")


## Docking tool Autodock, vina
class GenMolFile:
    def __init__(self, outpath, mgltools, mglbin):
        self.outpath = outpath
        self.prepare_ligand4 = os.path.join(
            mgltools, "AutoDockTools/Utilities24/prepare_ligand4.py"
        )
        self.mglbin = mglbin

        os.makedirs(os.path.join(outpath, "sdf"), exist_ok=True)
        os.makedirs(os.path.join(outpath, "mol2"), exist_ok=True)
        os.makedirs(os.path.join(outpath, "pdbqt"), exist_ok=True)

    def __call__(self, smi, mol_name, num_conf):
        sdf_file = os.path.join(self.outpath, "sdf", f"{mol_name}.sdf")
        mol2_file = os.path.join(self.outpath, "mol2", f"{mol_name}.mol2")
        pdbqt_file = os.path.join(self.outpath, "pdbqt", f"{mol_name}.pdbqt")

        mol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(mol)
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conf)
        for i in range(num_conf):
            AllChem.MMFFOptimizeMolecule(mol_h, confId=i)
        mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")

        # choose minimum energy conformer
        mi = np.argmin(
            [
                AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=i).CalcEnergy()
                for i in range(num_conf)
            ]
        )

        # write sdf
        print(Chem.MolToMolBlock(mol_h, confId=int(mi)), file=open(sdf_file, "w+"))

        # convert sdf to mol2
        os.system(f"obabel -isdf {sdf_file} -omol2 -O {mol2_file}")

        # convert ligand mol2 to ligand pdbqt
        os.system(
            f"{self.mglbin}/pythonsh {self.prepare_ligand4} -l {mol2_file} -o {pdbqt_file}"
        )
        return pdbqt_file


class DockVina_smi:
    def __init__(
        self,
        outpath,
        mgltools_dir=os.path.join("~/Docking/mgltools", "mgltools_x86_64Linux2_1.5.7"),
        vina_dir="~/Docking/qvina2/qvina",
        docksetup_dir=os.path.join("data", "seh/4jnc"),
        rec_file=None,
        bind_site=None,
        dock_pars="",
        cleanup=True,
    ):

        self.outpath = outpath
        self.mgltools = os.path.join(mgltools_dir, "MGLToolsPckgs")
        self.mgltools_bin = os.path.join(mgltools_dir, "bin")
        # self.vina_bin = os.path.join(vina_dir, "bin/vina")
        self.vina_bin = os.path.join(vina_dir, "bin/qvina02")

        self.rec_file = os.path.join(docksetup_dir, rec_file)
        self.bind_site = bind_site
        # self.config_file = config_file
        self.dock_pars = dock_pars
        self.cleanup = cleanup

        self.gen_molfile = GenMolFile(self.outpath, self.mgltools, self.mgltools_bin)
        # make vina command
        self.dock_cmd = (
            "{} --receptor {} "
            "--size_x {} --size_y {} --size_z {} "
            "--center_x {} --center_y {} --center_z {} "
            " --seed 0 --num_modes 9"  # default num_modes 9
            " --exhaustiveness 8 "  # default exhaustiveness 8
        )
        self.dock_cmd = self.dock_cmd.format(
            self.vina_bin, self.rec_file, *self.bind_site
        )
        self.dock_cmd += " --ligand {} --out {}"

        os.makedirs(os.path.join(self.outpath, "docked"), exist_ok=True)

    def dock(self, smi, mol_name=None, molgen_conf=20):
        mol_name = mol_name or "".join(
            random.choices(string.ascii_uppercase + string.digits, k=15)
        )
        docked_file = os.path.join(self.outpath, "docked", f"{mol_name}.pdbqt")
        input_file = self.gen_molfile(smi, mol_name, molgen_conf)

        # complete docking query
        dock_cmd = self.dock_cmd.format(input_file, docked_file)
        dock_cmd = dock_cmd + " " + self.dock_pars

        print(dock_cmd)

        # dock
        cl = subprocess.Popen(
            dock_cmd,
            shell=True,
            stdout=subprocess.PIPE,
        )
        cl.wait()

        # parse energy
        with open(docked_file) as f:
            docked_pdb = f.readlines()
        if docked_pdb[1].startswith("REMARK VINA RESULT"):
            dockscore = float(docked_pdb[1].split()[3])
        else:
            raise Exception("Can't parse docking energy")

        # parse coordinates
        # TODO: fix order
        coord = []
        endmodel_idx = 0
        for idx, line in enumerate(docked_pdb):
            if line.startswith("ENDMDL"):
                endmodel_idx = idx
                break

        docked_pdb_model_1 = docked_pdb[
            :endmodel_idx
        ]  # take only the model corresponding to the best energy
        docked_pdb_model_1_atoms = [
            line
            for line in docked_pdb_model_1
            if line.startswith("ATOM") and line.split()[2][0] != "H"
        ]  # ignore hydrogen
        coord.append([line.split()[-7:-4] for line in docked_pdb_model_1_atoms])
        coord = np.array(coord, dtype=np.float32)

        # convert out pdbqt to sdf
        os.system(
            f"obabel -ipdbqt {docked_file} -osdf -O {os.path.join(self.outpath, 'docked', f'{mol_name}.sdf')}"
        )

        if self.cleanup:
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(self.outpath, "sdf", f"{mol_name}.sdf"))
                os.remove(os.path.join(self.outpath, "mol2", f"{mol_name}.mol2"))
                os.remove(os.path.join(self.outpath, "pdbqt", f"{mol_name}.pdbqt"))
                os.remove(os.path.join(self.outpath, "docked", f"{mol_name}.pdb"))
        return mol_name, dockscore, coord


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_id", type=str, help="PDB ID")
    args = parser.parse_args()
    pdb_id = args.pdb_id

    df = pd.read_csv("../data/ligands/table2.csv")
    rslt_dir = f"../results/processed/lowest_conf/{pdb_id}"
    os.makedirs(rslt_dir, exist_ok=True)

    docksetup_dir = f"../data/receptors/"
    rec_file = f"{pdb_id}_processed.pdbqt"

    config_file = os.path.join(docksetup_dir, "grid_configs", f"{pdb_id}_processed")
    with open(config_file, "r") as f:
        grid_config = f.readlines()
    bind_site = [float(x.strip()) for x in grid_config]
    # print(f"Grid config: {grid_config}")

    # serialize docking
    wlines = []
    for i in range(df.shape[0]):
        # for i in range(5):
        smi = df["Smiles"][i]
        name = df["No."][i]
        dock = DockVina_smi(
            outpath=rslt_dir,
            docksetup_dir=docksetup_dir,
            rec_file=rec_file,
            bind_site=bind_site,
            cleanup=False,
        )

        mol_name, dockscore, coord = dock.dock(smi, mol_name=name)
        print(f"{mol_name}, {dockscore}")
        wlines.append(f"{mol_name}, {dockscore}")

    # parellel docking

    with open(os.path.join(rslt_dir, "docking_results.txt"), "w") as f:
        f.write("\n".join(wlines))
