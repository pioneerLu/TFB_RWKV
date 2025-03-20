python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon": 96}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 16, "base_T": 96, "batch_size": 16, "horizon": 96, "loss": "MSE", "lr": 0.005, "norm": true, "patience": 20, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ZafNoo/FITS"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon": 192}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 16, "base_T": 96, "batch_size": 16, "horizon": 192, "loss": "MSE", "lr": 0.005, "norm": true, "patience": 20, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ZafNoo/FITS"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon": 336}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 16, "base_T": 96, "batch_size": 16, "horizon": 336, "loss": "MSE", "lr": 0.005, "norm": true, "patience": 20, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ZafNoo/FITS"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon": 720}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 14, "base_T": 96, "batch_size": 64, "horizon": 720, "loss": "MSE", "lr": 0.05, "norm": true, "patience": 20, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ZafNoo/FITS"

