use maestro::prelude::*;
use std::{ffi::OsStr, path::Path};

#[maestro::main]
fn main() {
    let input_files = arg!("input_files").split(',').map(Path::new).map(|path| {
        (
            path,
            path.file_stem()
                .expect("All paths in input_files should have a resolveable filename!"),
        )
    });
    for (path, _) in input_files.clone() {
        assert!(path.exists(), "All paths in input_files must exist!")
    }

    let workflow = |(path, name): (&Path, &OsStr)| -> Result<PathBuf, io::Error> {
        let molecule_name = &name.to_string_lossy();
        let [prmtop, inpcrd] = tleap(path, molecule_name)?.into_array();
        let [gro, topol] = parmed(prmtop, inpcrd, molecule_name)?.into_array();
        let [gromacs_workdir] = gromacs(gro, topol, molecule_name)?.into_array();
        Ok(gromacs_workdir)
    };

    let process_results = parallelize(input_files, workflow)
        .into_iter()
        .map(|result| result.unwrap());
    println!("All processes terminated! Outputs: {process_results:#?}");
}

fn tleap(input: &Path, molecule_name: &str) -> WorkflowResult {
    let prmtop = Path::new("system.prmtop");
    let inpcrd = Path::new("system.inpcrd");
    let solvated_pdb = Path::new("system_solvated.pdb");

    workflow! {
        name = format!("tleap_{molecule_name}"),
        executor = "direct",
        inputs = [input],
        outputs = [prmtop, inpcrd, solvated_pdb],
        dependencies = ["!", "tleap"],
        process = r#"
        echo "
            source leaprc.water.tip3p
            source leaprc.protein.ff14SB
            mol = loadpdb $input
            solvateBox mol TIP3PBOX 10
            addions mol Na+ 0
            addions mol Cl- 0
            saveamberparm mol $prmtop $inpcrd
            savepdb mol $solvated_pdb
            quit
        " > tleap.in
        tleap -f tleap.in > tleap.log
        "#
    }
}

fn parmed(prmtop: PathBuf, inpcrd: PathBuf, molecule_name: &str) -> WorkflowResult {
    let gro = Path::new("system.gro");
    let topol = Path::new("topol.top");

    workflow! {
        name = format!("parmed_{molecule_name}"),
        executor = "direct",
        inputs = [prmtop, inpcrd],
        outputs = [gro, topol],
        dependencies = ["!", "python", "py:parmed"],
        process = r#"
            echo "
                import parmed as pmd
                parm = pmd.load_file('$prmtop', '$inpcrd')

                parm.save('$gro', format='gro')
                parm.save('$topol', format='gromacs')
            " > parmed_convert.py
            python parmed_convert.py
        "#
    }
}

fn gromacs(gro: PathBuf, topol: PathBuf, molecule_name: &str) -> WorkflowResult {
    workflow! {
        name = format!("gromacs_{molecule_name}"),
        executor = "slurm",
        inputs = [gro, topol],
        process = r#"
            set -euo pipefail

            echo "===> Setting environment variables!"
            PATH=/arc/project/st-shallam-1/igem-2025/molecular-dynamics/.pixi/envs/default/bin/:$PATH
            export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

            echo "===> Step 1: Energy Minimization"
            gmx grompp -f minim.mdp -c system.gro -p topol.top -o em.tpr -maxwarn 1
            gmx mdrun -deffnm em -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -nb gpu -pin on

            echo "===> Step 2: NVT Equilibration"
            gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 1
            gmx mdrun -deffnm nvt -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -nb gpu -pin on

            echo "===> Step 3: NPT Equilibration"
            gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 1
            gmx mdrun -deffnm npt -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -nb gpu -pin on

            echo "===> Step 4: Production MD"
            gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 1
            gmx mdrun -deffnm md -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -nb gpu -pin on

            echo "===> MD Simulation Complete"
        "#
    }
}
