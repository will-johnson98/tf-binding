#!/bin/bash
# SGE job script for running reprocess.py with optimized memory settings
# Submit with: qsub run_reprocess_sge.sh

# SGE directives
#$ -N reprocess_job                    # Job name
#$ -cwd                               # Run in current working directory
#$ -j y                               # Merge stdout and stderr
#$ -o logs/reprocess_$JOB_ID.log       # Output log file
#$ -e logs/reprocess_$JOB_ID.err       # Error log file

# Enhanced resource requirements for memory-intensive tasks
#$ -pe smp 2                          # Request 2 cores (reduce cores, increase memory per core)
#$ -l h_vmem=32G                      # Request 16GB virtual memory per core
#$ -l mem_free=12G                    # Request 12GB free memory per core  
#$ -l h_rt=48:00:00                   # Request 48 hours runtime (increased)

# Optional: Request specific node types with more memory
# #$ -l mem_total=64G                 # Total memory on node

# Create logs directory if it doesn't exist
mkdir -p logs

# Set memory-related environment variables for Python/Polars
export POLARS_MAX_THREADS=2           # Limit Polars threads to match SGE cores
export OMP_NUM_THREADS=2              # Limit OpenMP threads
export MKL_NUM_THREADS=2              # Limit MKL threads if using Intel MKL

# Set up environment
source ~/.bashrc

# Activate conda environment
conda activate pterodactyl

# Function to log system resources
log_resources() {
    echo "=== System Resources ==="
    echo "Available Memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
    echo "CPU Cores: $(nproc)"
    echo "Load Average: $(uptime | awk -F'load average:' '{ print $2 }')"
    echo "========================"
}

# Function to monitor memory usage
monitor_memory() {
    echo "=== Memory Usage ==="
    ps -p $1 -o pid,ppid,cmd,%mem,%cpu --no-headers
    echo "==================="
}

# Print job information
echo "=== Job Information ==="
echo "Job ID: $JOB_ID"
echo "Job Name: $JOB_NAME" 
echo "Host: $HOSTNAME"
echo "Working Directory: $PWD"
echo "Start Time: $(date)"
echo "Allocated Cores: $NSLOTS"
echo "========================="

# Log initial resources
log_resources

# Run the Python script with memory monitoring
echo "Starting reprocess.py..."

# Start the Python process in background and get PID
python reprocess.py &
PYTHON_PID=$!

# Monitor memory usage every 5 minutes
(
    while kill -0 $PYTHON_PID 2>/dev/null; do
        sleep 300  # Wait 5 minutes
        monitor_memory $PYTHON_PID
    done
) &
MONITOR_PID=$!

# Wait for Python process to complete
wait $PYTHON_PID
EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true

# Final resource check
log_resources

# Check exit status
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully at $(date)"
else
    echo "Job failed with exit code $EXIT_CODE at $(date)"
    
    # Additional debugging info on failure
    echo "=== Debug Information ==="
    echo "Final memory state:"
    free -h
    echo "Disk space:"
    df -h .
    echo "========================="
    
    exit $EXIT_CODE
fi

echo "=== Job Summary ==="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "==================="