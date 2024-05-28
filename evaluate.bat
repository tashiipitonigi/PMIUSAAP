python evaluate_1.py -c "denoise_model.ckpt" -i "Results/Evaluation/ped/int_80/10/" -n 0.010
conda run -n pytorch3d python ./evaluate_2.py -i "Results/Evaluation/ped/int_80/10"