python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.DLinear" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "factor": 3, "horizon": 96, "norm": true, "seq_len": 512}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh2/DLinear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 192}' --model-name "time_series_library.DLinear" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "factor": 3, "horizon": 192, "norm": true, "seq_len": 336}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh2/DLinear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.DLinear" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "factor": 3, "horizon": 336, "norm": true, "seq_len": 336}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh2/DLinear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 32, "d_ff": 2048, "d_model": 512, "horizon": 720, "lr": 0.005, "norm": true, "seq_len": 336}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh2/DLinear"

