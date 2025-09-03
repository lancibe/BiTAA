#!/bin/bash

# Function to check the current conda environment
check_conda_env() {
    if [[ "$CONDA_DEFAULT_ENV" != "lgm" ]]; then
        echo "[INFO] Activating conda environment 'lgm'..."
        source activate lgm  # Adjust path as needed for your conda installation
    # else
    #     echo "Already in conda environment 'lgm'."
    fi
}

# Function to find a GPU with volatile utilization of 0 and memory usage below 10%
find_available_gpu() {
    echo "[INFO] Checking available GPUs..."
    total_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    gpu_id=-1
    for i in $(seq 0 $((total_gpus - 1))); do
        utilization=$(nvidia-smi -i $i --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        memory_usage=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
        memory_total=$(nvidia-smi -i $i --query-gpu=memory.total --format=csv,noheader,nounits)
        memory_percentage=$((memory_usage * 100 / memory_total))
        
        if [[ $utilization -eq 0 && $memory_percentage -lt 10 ]]; then
            gpu_id=$i
            break
        fi
    done
    
    if [[ $gpu_id -ge 0 ]]; then
        echo "[INFO] Selected GPU: $gpu_id (utilization: $utilization%, memory usage: $memory_percentage%)"
        export CUDA_VISIBLE_DEVICES=$gpu_id
    else
        echo -e "\e[31m[ERROR] No available GPU with utilization 0 and memory usage below 10% found. Exiting.\e[0m"
        exit 1
    fi
}



# Main script execution
check_conda_env
find_available_gpu
# current_date=$(date +'%m%d%H%M')
current_date=$(date +'%m%d')
# echo "Current date in MM-DD format: $current_date"
# vehicle=vehicle.tesla.cybertruck
# vehicle=vehicle.toyota.prius
# vehicle=vehicle.mini.cooper_s
# vehicle=vehicle.seat.leon
vehicle=vehicle.audi.a2 # first choice
# vehicle=vehicle.nissan.micra
# vehicle=vehicle.micro.microlino
# vehicle=vehicle.citroen.c3
# vehicle.toyota.prius, vehicle.mini.cooper_s, vehicle.tesla.cybertruck, vehicle.seat.leon
# vehicle.nissan.micra, vehicle.micro.microlino, vehicle.citroen.c3, vehicle.audi.a2

python infer_test_2.py big --resume pretrained/model_fp16_fixrot.safetensors --workspace workspace/$current_date --test_path core/carla_dataset_full/$vehicle/0

echo "[INFO] Done Generation in workspace/$current_date!"

python attack_test.py big --workspace workspace/$current_date


# Combine two videos side by side using ffmpeg
video1="workspace/$current_date/0.mp4"
video2="workspace/$current_date/0_attack.mp4"
output_video="workspace/$current_date/combined_video.mp4"

echo "[INFO] Combining videos $video1 and $video2 into $output_video..."

ffmpeg -i $video1 -i $video2 -filter_complex hstack $output_video -loglevel quiet -y

echo "[INFO] Combined video saved as $output_video!"