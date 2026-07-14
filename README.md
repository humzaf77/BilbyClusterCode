So, I didn't get these issues running on my computer, which means the error is almost entirely from the way the environments are built and transferred between our machines. I think the best thing to do is to rebuild the 

source /resnick/groups/RelativityTheory/alaeuger/miniconda3/etc/profile.d/conda.sh
conda config --set channel_priority strict
conda env remove -n BilbyEnv
conda env create -f environment_AL.yml
conda activate BilbyEnv
python -c "import lal, lalsimulation, bilby, gwpy, pycbc; print('OK')"

then maybe you could check for one event with sbatch run_catalog15_AL.slurm GW170104, and if it says "imports ok" we can submit_all_AL.sh
