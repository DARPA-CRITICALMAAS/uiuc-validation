#!/bin/bash
pred_data="../inference_pipeline/results/golden_muscat/final_evaluation/output"
true_data="/projects/bcxi/shared/datasets/final_evaluation"
results="results/golden_muscat/final_evaluation"
processes=64
job_name="usgs_validation"

mkdir -p "batch_scripts"
cat <<EOF > "batch_scripts/${job_name}.sh"
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/slurm/${job_name}_%j.o
#SBATCH --error=logs/slurm/${job_name}_%j.e

#SBATCH --account=bcxi-tgirails
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00

# Job Commands
echo "Running usgs validation on ${pred_data}"
echo "Job ${jobname}:$SLURM_JOB_ID running on $(hostname)"

# Activate conda environment
conda activate cmass_validation

python usgsDemo.py -p ${pred_data} -t ${true_data}/true_segmentations -m ${true_data}/images -l ${true_data}/legends -o ${results} --processes ${processes} --log logs/$(basename ${pred_data})_%j.log

echo "Job ${jobname}:$SLURM_JOB_ID finished running on $(hostname)"
EOF

# Submit job
sbatch "batch_scripts/${job_name}.sh"
echo "Submitted job ${job_name}"
