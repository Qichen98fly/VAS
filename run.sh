#!/bin/bash
#SBATCH --job-name=showui2b_infer
#SBATCH --partition=cybersecurity
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G

export PYTHONPATH=/opt/home/s4184496/project/VAS:$PYTHONPATH

# python src/eval/run_llama2.py \
#   --model-path liuhaotian/llava-v1.5-7b \
#   --image-file ./C_datasets/YOUR_DATASET/000000084664.jpg \
#   --query "Is there a egg?" \
#   --conv-mode llava_v1 \
#   --output-dir ./results/visualizations_7b \
#   --vis-layers 31 \
#   --tau 20 \
#   --dim-sink 2533 1415 \
#   --resolution 256 \
#   --temperature 0.2 \
#   --max_new_tokens 128


# python src/eval/run_llava.py \
#   --model-path liuhaotian/llava-v1.5-7b \
#   --image-file ./C_datasets/YOUR_DATASET/000000079565.jpg \
#   --query "Is there a giraffe in the image?" \
#   --sep "," \
#   --conv-mode llava_v1 \
#   --temperature 0.2 \
#   --max_new_tokens 256

python src/eval/run_llava.py \
  --model-path liuhaotian/llava-v1.5-13b \
  --image-file ./C_datasets/YOUR_DATASET/000000079565.jpg \
  --query "Is there a giraffe in the image?" \
  --sep "," \
  --conv-mode llava_v1 \
  --temperature 0.2 \
  --max_new_tokens 256

