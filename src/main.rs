use maestro::prelude::*;

#[maestro::main]
fn main() {
    let init_msg: &str = arg!("init_msg");
    let input_files: &[&Path] = inputs!("input_files");

    println!("{init_msg}");

    let workflow = |path: &&Path| -> NodeResult<PathBuf> {
        let molecule_name = &path
            .file_stem()
            .expect("All paths in input_files should have a resolveable filename!")
            .to_string_lossy();
        println!("Started workflow {molecule_name:?}");

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

    process! {
        /// Runs tleap to prepare input files for MD simulation
        name = format!("tleap_{molecule_name}"),
        executor = "direct",
        inputs = [input],
        outputs = [prmtop, inpcrd, solvated_pdb],
        dependencies = ["!", "tleap"],
        script = r#"
            PATH=/arc/project/st-shallam-1/igem-2025/molecular-dynamics/.pixi/envs/default/bin/:$PATH
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

    process! {
        /// Executes parmed to convert from AMBER into a GROMACS-compatible format
        name = format!("parmed_{molecule_name}"),
        executor = "direct",
        inputs = [prmtop, inpcrd],
        outputs = [gro, topol],
        dependencies = ["!", "python", "py:parmed"],
        script = r#"
            PATH=/arc/project/st-shallam-1/igem-2025/molecular-dynamics/.pixi/envs/default/bin/:$PATH
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
    let minim_mdp = Path::new("data/minim.mdp");
    let nvt_mdp = Path::new("data/nvt.mdp");
    let npt_mdp = Path::new("data/npt.mdp");
    let md_mdp = Path::new("data/md.mdp");

    process! {
        /// Executes the primary GROMACS molecular dynamics process
        name = format!("gromacs_{molecule_name}"),
        executor = "slurm",
        inputs = [
            gro,
            topol,
            minim_mdp,
            nvt_mdp,
            npt_mdp,
            md_mdp,
        ],
        script = r#"
            set -euo pipefail

            echo "===> Setting environment variables!"
            PATH=/arc/project/st-shallam-1/igem-2025/molecular-dynamics/.pixi/envs/default/bin/:$PATH
            export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

            echo "===> Step 1: Energy Minimization"
            gmx grompp -f "$minim_mdp" -c "$gro" -p "$topol" -o em.tpr -maxwarn 1
            gmx mdrun -deffnm em -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -nb gpu -pin on

            echo "===> Step 2: NVT Equilibration"
            gmx grompp -f "$nvt_mdp" -c em.gro -r em.gro -p "$topol" -o nvt.tpr -maxwarn 1
            gmx mdrun -deffnm nvt -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -nb gpu -pin on

            echo "===> Step 3: NPT Equilibration"
            gmx grompp -f "$npt_mdp" -c nvt.gro -r nvt.gro -t nvt.cpt -p "$topol" -o npt.tpr -maxwarn 1
            gmx mdrun -deffnm npt -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -nb gpu -pin on

            echo "===> Step 4: Production MD"
            gmx grompp -f "$md_mdp" -c npt.gro -t npt.cpt -p "$topol" -o md.tpr -maxwarn 1
            gmx mdrun -deffnm md -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -nb gpu -pin on

            echo "===> MD Simulation Complete"
        "#
    }
}
