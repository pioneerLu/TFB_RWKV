python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS-BAY.csv" --strategy-args '{"horizon": 96}' --model-name "pdf.PDF" --model-hyper-params '{"batch_size": 64, "d_ff": 256, "d_model": 128, "dropout": 0.35, "e_layers": 3, "fc_dropout": 0.15, "horizon": 96, "kernel_list": [3, 7, 9, 11], "n_head": 32, "norm": true, "patch_len": [1], "patience": 5, "pc_start": 0.2, "period": [24], "seq_len": 512, "stride": [1], "train_epochs": 20}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS-BAY/PDF"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS-BAY.csv" --strategy-args '{"horizon": 192}' --model-name "pdf.PDF" --model-hyper-params '{"batch_size": 64, "d_ff": 256, "d_model": 128, "dropout": 0.35, "e_layers": 3, "fc_dropout": 0.15, "horizon": 192, "kernel_list": [3, 7, 9, 11], "n_head": 32, "norm": true, "patch_len": [1], "patience": 5, "pc_start": 0.2, "period": [24], "seq_len": 512, "stride": [1], "train_epochs": 20}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS-BAY/PDF"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS-BAY.csv" --strategy-args '{"horizon": 336}' --model-name "pdf.PDF" --model-hyper-params '{"batch_size": 64, "d_ff": 256, "d_model": 128, "dropout": 0.35, "e_layers": 3, "fc_dropout": 0.15, "horizon": 336, "kernel_list": [3, 7, 9, 11], "n_head": 32, "norm": true, "patch_len": [1], "patience": 5, "pc_start": 0.2, "period": [24], "seq_len": 512, "stride": [1], "train_epochs": 20}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS-BAY/PDF"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS-BAY.csv" --strategy-args '{"horizon": 720}' --model-name "pdf.PDF" --model-hyper-params '{"batch_size": 64, "d_ff": 256, "d_model": 128, "dropout": 0.35, "e_layers": 3, "fc_dropout": 0.15, "horizon": 720, "kernel_list": [3, 7, 9, 11], "n_head": 32, "norm": true, "patch_len": [1], "patience": 5, "pc_start": 0.2, "period": [24], "seq_len": 512, "stride": [1], "train_epochs": 20}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS-BAY/PDF"

