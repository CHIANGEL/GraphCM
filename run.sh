python -u run.py --train --optim adam --eval_freq 100 --check_point 100 \
--dataset demo --combine exp_mul \
--gnn_neigh_sample 0 --gnn_concat False --inter_neigh_sample 0 \
--learning_rate 0.001 --lr_decay 0.5 --weight_decay 1e-5 --dropout_rate 0.5 \
--num_steps 20000 --embed_size 64 --hidden_size 64 --batch_size 256 --patience 6 \
--model_dir ./GraphCM/models/ \
--result_dir ./GraphCM/results/ \
--summary_dir ./GraphCM/summary/ \
--log_dir ./GraphCM/log/