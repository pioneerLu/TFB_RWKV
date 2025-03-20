python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon": 96}' --model-name "pdf.PDF" --model-hyper-params '{"batch_size": 64, "d_ff": 128, "d_model": 64, "dropout": 0.5, "e_layers": 3, "fc_dropout": 0.2, "horizon": 96, "kernel_list": [3, 11, 15, 23, 27], "learning_rate": 0.00015, "n_head": 16, "norm": true, "patch_len": [16, 16, 24], "patience": 20, "period": [144, 180, 720], "seq_len": 512, "stride": [16, 16, 24], "train_epochs": 100}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Weather/PDF"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon": 192}' --model-name "pdf.PDF" --model-hyper-params '{"batch_size": 64, "d_ff": 128, "d_model": 64, "dropout": 0.5, "e_layers": 3, "fc_dropout": 0.2, "horizon": 192, "kernel_list": [3, 11, 15, 23, 27], "learning_rate": 0.00015, "n_head": 16, "norm": true, "patch_len": [16, 16, 24], "patience": 20, "period": [144, 180, 720], "seq_len": 512, "stride": [16, 16, 24], "train_epochs": 100}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Weather/PDF"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon": 336}' --model-name "pdf.PDF" --model-hyper-params '{"batch_size": 64, "d_ff": 256, "d_model": 128, "dropout": 0.35, "e_layers": 3, "fc_dropout": 0.15, "horizon": 336, "kernel_list": [3, 7, 9, 11], "n_head": 32, "norm": true, "patch_len": [1], "patience": 5, "pc_start": 0.2, "period": [24], "seq_len": 512, "stride": [1], "train_epochs": 20}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Weather/PDF"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon": 720}' --model-name "pdf.PDF" --model-hyper-params '{"batch_size": 64, "d_ff": 256, "d_model": 128, "dropout": 0.35, "e_layers": 3, "fc_dropout": 0.15, "horizon": 720, "kernel_list": [3, 7, 9, 11], "n_head": 32, "norm": true, "patch_len": [1], "patience": 5, "pc_start": 0.2, "period": [24], "seq_len": 336, "stride": [1], "train_epochs": 20}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Weather/PDF"

