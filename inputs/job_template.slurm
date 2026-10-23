#!/bin/bash
# Slurm job template for diagramc: one Markov chain per MPI rank, one rank per physical core.
# Written for nodes with 2 sockets x 24 cores (48 ranks per node); adapt the ADAPT lines and the paths.
#
# Usage: set the ADAPT lines (modules, paths) and, if needed, the ##SBATCH partition/qos lines; copy into the run
# directory next to input.json (or a template.json with "seed": SEED, see below) and
#     sbatch job_template.slurm
#
# Checks before a production job (see inputs/README.md):
#   - params.max_time (default 84600 s = 23 h 30 min) is the budget of the WHOLE job: keep --time about 30 min above
#     it (startup and output writing), e.g. --time=24:00:00 with the default
#   - config.rng.seed different for every independent job (the ranks within a job differ automatically)
#   - output.prefix unique per job
#   - start without external lines (n_external_max = 0): markdown_notes/external_line_traps.md

#SBATCH --job-name=dmc
#SBATCH --nodes=1                    # 48 chains per node; more nodes = more chains (each pays its thermalization)
#SBATCH --ntasks-per-node=48         # one rank per physical core: 2 sockets x 24 cores
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread         # physical cores only, no hyperthread siblings
#SBATCH --time=1-00:00:00              # > params.max_time + ~30 min
##SBATCH --partition=PARTITION       # cluster-specific: remove one '#' and set it
##SBATCH --qos=QOS                   # cluster-specific: remove one '#' and set it
#SBATCH --output=%x_%j.log

# the same modules (and versions) used to build bin/diagramc: GCC >= 13 (for libstdc++) and the MPI library
GCC_MODULE="gcc/13.x"                # ADAPT: e.g. the output of `module list` saved at build time
MPI_MODULE="intel-mpi/2021.x"        # ADAPT
module purge
module load "$GCC_MODULE" "$MPI_MODULE"

DMC=$HOME/DiagMC/DMC_Trace           # ADAPT: repository with bin/diagramc
RUN=$SLURM_SUBMIT_DIR                # run directory: input.json here, outputs under output.prefix
cd "$RUN" || exit 1

# optional: input.json from a template whose seed line reads   "seed": SEED   (one seed per job, from the job id;
# keep seeds below 2^53 if other tools touch the file)
if [ -f template.json ] && [ ! -f input.json ]; then
    sed "s/\"seed\": SEED/\"seed\": $SLURM_JOB_ID/" template.json > input.json
fi

# every shared library must resolve with the loaded modules ("not found" = a module is missing)
ldd "$DMC/bin/diagramc" | grep -E "not found" && { echo "missing libraries: check the modules"; exit 1; }
module list 2>&1

export OMP_NUM_THREADS=1             # the code is not threaded

mpirun -np "$SLURM_NTASKS" "$DMC/bin/diagramc" input.json
# Intel MPI under srun may need the Slurm PMI library (see the cluster documentation), e.g.
#   export I_MPI_PMI_LIBRARY=/path/to/libpmi2.so
# if all ranks report "1 chain(s)", MPI is not connected to Slurm: set it, or use the mpirun line instead

# srun --cpu-bind=cores "$DMC/bin/diagramc" input.json 
status=$?

# 0: ok; 2: an estimator could not be written; 3: production skipped (time budget used up by the thermalization)
# to continue the run in a new job: set "checkpoint": {"restart": true} in input.json and resubmit this script from
# the same directory with the same number of ranks (see inputs/README.md, config.checkpoint)
echo "diagramc exit code $status"
exit $status
