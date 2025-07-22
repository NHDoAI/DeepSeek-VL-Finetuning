#!/bin/bash
cd "$(dirname "$0")"
python3 ./full_b6-s23_short-lidar_focal-loss_custom-alpha_2epochs_best/combined_model_evaluation_no-sent_short-prompt_fixed-eos.py
echo "s23_best finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s23_short-lidar_focal-loss_custom-alpha_2epochs/best_chkpoint/merged_model/
echo "--------------------------------"

python3 ./full_b6-s23_short-lidar_focal-loss_custom-alpha_2epochs_full-epoch/combined_model_evaluation_no-sent_short-prompt_fixed-eos.py
echo "s23_epoch-1 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s23_short-lidar_focal-loss_custom-alpha_2epochs/first_epoch_chkpoint/merged_model/
echo "--------------------------------"

python3 ./full_b6-s23_short-lidar_focal-loss_custom-alpha_2epochs_step2299/combined_model_evaluation_no-sent_short-prompt_fixed-eos.py
echo "s23_step2299 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s23_short-lidar_focal-loss_custom-alpha_2epochs/manual_eval_chkpoint_30/merged_model/
echo "--------------------------------"

python3 ./full_b6-s23_short-lidar_focal-loss_custom-alpha_2epochs_epoch1.5/combined_model_evaluation_no-sent_short-prompt_fixed-eos.py
echo "s23_epoch-1.5 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s23_short-lidar_focal-loss_custom-alpha_2epochs/manual_eval_chkpoint_61/merged_model/
echo "--------------------------------"

python3 ./full_b6-s23_short-lidar_focal-loss_custom-alpha_2epochs_epoch2/combined_model_evaluation_no-sent_short-prompt_fixed-eos.py
echo "s23_epoch-2 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s23_short-lidar_focal-loss_custom-alpha_2epochs/manual_eval_chkpoint_86/merged_model/
echo "--------------------------------"

python3 ./full_b6-s42_short-lidar_focal-loss_custom-alpha_2epochs_best/combined_model_evaluation_no-sent_short-prompt_fixed-eos.py
echo "s42_best finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s42_short-lidar_focal-loss_custom-alpha_2epochs/best_chkpoint/merged_model/
echo "--------------------------------"

python3 ./full_b6-s42_short-lidar_focal-loss_custom-alpha_2epochs_full-epoch/combined_model_evaluation_no-sent_short-prompt_fixed-eos.py
echo "s42_epoch-1 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s42_short-lidar_focal-loss_custom-alpha_2epochs/first_epoch_chkpoint/merged_model/
echo "--------------------------------"

python3 ./full_b6-s42_short-lidar_focal-loss_custom-alpha_2epochs_step2299/combined_model_evaluation_no-sent_short-prompt_fixed-eos.py
echo "s42_step2299 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s42_short-lidar_focal-loss_custom-alpha_2epochs/manual_eval_chkpoint_30/merged_model/
echo "--------------------------------"

python3 ./full_b6-s42_short-lidar_focal-loss_custom-alpha_2epochs_epoch1.5/combined_model_evaluation_no-sent_short-prompt_fixed-eos.py
echo "s42_epoch-1.5 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s42_short-lidar_focal-loss_custom-alpha_2epochs/manual_eval_chkpoint_61/merged_model/
echo "--------------------------------"

python3 ./full_b6-s42_short-lidar_focal-loss_custom-alpha_2epochs_epoch2/combined_model_evaluation_no-sent_short-prompt_fixed-eos.py
echo "s42_epoch-2 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s42_short-lidar_focal-loss_custom-alpha_2epochs/manual_eval_chkpoint_86/merged_model/
echo "--------------------------------"

python3 ./full_b6-s322_short-lidar_focal-loss_custom-alpha_2epochs_best/combined_model_evaluation_no-sent_short-prompt_fixed-eos.py
echo "s322_best finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s322_short-lidar_focal-loss_custom-alpha_2epochs/best_chkpoint/merged_model/
echo "--------------------------------"

python3 ./full_b6-s322_short-lidar_focal-loss_custom-alpha_2epochs_full-epoch/combined_model_evaluation_no-sent_short-prompt_fixed-eos.py
echo "s322_epoch-1 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s322_short-lidar_focal-loss_custom-alpha_2epochs/first_epoch_chkpoint/merged_model/
echo "--------------------------------"

python3 ./full_b6-s322_short-lidar_focal-loss_custom-alpha_2epochs_step2299/combined_model_evaluation_no-sent_short-prompt_fixed-eos.py
echo "s322_step2299 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s322_short-lidar_focal-loss_custom-alpha_2epochs/manual_eval_chkpoint_30/merged_model/
echo "--------------------------------"

python3 ./full_b6-s322_short-lidar_focal-loss_custom-alpha_2epochs_epoch1.5/combined_model_evaluation_no-sent_short-prompt_fixed-eos.py
echo "s322_epoch-1.5 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s322_short-lidar_focal-loss_custom-alpha_2epochs/manual_eval_chkpoint_61/merged_model/
echo "--------------------------------"

python3 ./full_b6-s322_short-lidar_focal-loss_custom-alpha_2epochs_epoch2/combined_model_evaluation_no-sent_short-prompt_fixed-eos.py
echo "s322_epoch-2 finished"
echo "removing merged model"
rm -r /home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s322_short-lidar_focal-loss_custom-alpha_2epochs/manual_eval_chkpoint_86/merged_model/
echo "--------------------------------"