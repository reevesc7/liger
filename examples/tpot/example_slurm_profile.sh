#SBATCH --time=00:30:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

# Set to path of your environment activation script or global activation command
# (e.g., conda activate <env name>)
source /home/reeves/documents/schossau/liger/.venv/bin/activate
