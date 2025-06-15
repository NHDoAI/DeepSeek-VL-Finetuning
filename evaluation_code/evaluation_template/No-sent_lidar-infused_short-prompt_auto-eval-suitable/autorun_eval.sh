#!/bin/bash

python3 ./Short-prompt-v2-short-lidar_b6_seed322_loss_epoch1/combined_model_evaluation_no-sent.py
echo "s322_epoch1 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/Short-prompt-v2-short-lidar_b6-seed322_loss/first_epoch_chkpoint/merged_model/
echo "--------------------------------"

python3 ./Short-prompt-v2-short-lidar_b6_seed322_loss_best/combined_model_evaluation_no-sent.py
echo "s322_best finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/Short-prompt-v2-short-lidar_b6-seed322_loss/best_chkpoint/merged_model/
echo "--------------------------------"

python3 ./Short-prompt-v2-short-lidar_b6_seed322_loss_step1349/combined_model_evaluation_no-sent.py
echo "s322_step1349 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/Short-prompt-v2-short-lidar_b6-seed322_loss/manual_eval_chkpoint_11/merged_model/
echo "--------------------------------"

python3 ./Short-prompt-v2-short-lidar_b6_seed42_loss_epoch1/combined_model_evaluation_no-sent.py
echo "s42_epoch1 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/Short-prompt-v2-short-lidar_b6-seed42_loss/first_epoch_chkpoint/merged_model/
echo "--------------------------------"

python3 ./Short-prompt-v2-short-lidar_b6_seed42_loss_best/combined_model_evaluation_no-sent.py
echo "s42_best finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/Short-prompt-v2-short-lidar_b6-seed42_loss/best_chkpoint/merged_model/
echo "--------------------------------"

python3 ./Short-prompt-v2-short-lidar_b6_seed42_loss_step1349/combined_model_evaluation_no-sent.py
echo "s42_step1349 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/Short-prompt-v2-short-lidar_b6-seed42_loss/manual_eval_chkpoint_11/merged_model/
echo "--------------------------------"

python3 ./Short-prompt-v2-short-lidar_b6_seed23_loss_epoch1/combined_model_evaluation_no-sent.py
echo "s23_epoch1 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/Short-prompt-v2-short-lidar_b6-seed23_loss/first_epoch_chkpoint/merged_model/
echo "--------------------------------"

python3 ./Short-prompt-v2-short-lidar_b6_seed23_loss_best/combined_model_evaluation_no-sent.py
echo "s23_best finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/Short-prompt-v2-short-lidar_b6-seed23_loss/best_chkpoint/merged_model/
echo "--------------------------------"

python3 ./Short-prompt-v2-short-lidar_b6_seed23_loss_step1349/combined_model_evaluation_no-sent.py
echo "s23_step1349 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/Short-prompt-v2-short-lidar_b6-seed23_loss/manual_eval_chkpoint_11/merged_model/
echo "--------------------------------"