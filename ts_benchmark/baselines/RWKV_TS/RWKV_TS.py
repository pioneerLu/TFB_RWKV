import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ...models.model_base import ModelBase
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
# import pdb
# pdb.set_trace()
from .model import UniversalRWKVTimeSeries
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn

from torch import optim
from ts_benchmark.models.model_base import ModelBase
from sklearn.preprocessing import StandardScaler
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
    get_time_mark
)
from ts_benchmark.utils.data_processing import split_before
from ts_benchmark.baselines.duet.utils.tools import EarlyStopping, adjust_learning_rate
import os

DEFAULT_PARAMS = {
    "load_model": "",
    "wandb": "",
    "proj_dir": "out",
    "run_name": "demo_run",
    "random_seed": -1,
    "data_file": "",
    "data_type": "utf-8",
    "vocab_size": 0,
    "ctx_len": 100,
    "epoch_steps": 1000,
    "epoch_count": 50,
    "epoch_begin": 0,
    "epoch_save": 5,
    "micro_bsz": 12,
    "n_layer": 6,
    "n_embd": 512,
    "dim_att": 0,
    "dim_ffn": 0,
    "pre_ffn": 0,
    "head_size_a": 64,
    "head_size_divisor": 8,
    "lr_init": 6e-4,
    "lr_final": 1e-5,
    "warmup_steps": -1,
    "beta1": 0.9,
    "beta2": 0.99,
    "adam_eps": 1e-8,
    "grad_cp": 0,
    "dropout": 0.0,
    "weight_decay": 0.0,
    "weight_decay_final": -1.0,
    "ds_bucket_mb": 200,
    "n_emb_layer": 4,
    "sma_window": 3,
    "validate_only": 0,
    'batch_size':64,
    'num_workers':0,
    'loss':'MSE',
    'lr':1e-5,
    "patience": 10,
    "num_epochs": 20,
    'precision':'bf16',
    'lradj':"type3",
}

class RWKVConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)
        if self.dim_att <= 0:
            self.dim_att = self.n_embd
        if self.batch_size <= 0 or self.batch_size is None:
            self.batch_size = self.micro_bsz
        if self.lr is None:
            self.lr = self.lr_init
        if self.precision is None:
            self.precision = 'bf16'
    @property
    def pred_len(self):
        return self.horizon


class RWKV_TS(ModelBase):
    def __init__(self, **kwargs):
        self.scaler = StandardScaler()
        self.config = RWKVConfig(**kwargs)
        self.results = None
    
        # os.environ["RWKV_CTXLEN"] = str(self.config.ctx_len)
        # os.environ["RWKV_HEAD_SIZE_A"] = str(self.config.head_size_a)
        # os.environ["RWKV_FLOAT_MODE"] = self.config.precision
        # os.environ["RWKV_JIT_ON"] = "1"
        # if "deepspeed_stage_3" in self.config.strategy:
        #     os.environ["RWKV_JIT_ON"] = "0"
    @property
    def model_name(self):
        """
        Returns the name of the model.
        """
        return "RWKV_TS"
    def multi_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        if self.model_name == "MICN":
            setattr(self.config, "label_len", self.config.seq_len)
        else:
            setattr(self.config, "label_len", self.config.seq_len // 2)

    def single_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        setattr(self.config, "label_len", self.config.horizon)
    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by VAR.

        :return: An empty dictionary indicating that VAR does not require additional hyperparameters.
        """
        return {
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm"
        }

    def padding_data_for_forecast(self, test):
        time_column_data = test.index
        data_colums = test.columns
        start = time_column_data[-1]
        # padding_zero = [0] * (self.config.horizon + 1)
        date = pd.date_range(
            start=start, periods=self.config.horizon + 1, freq=self.config.freq.upper()
        )
        df = pd.DataFrame(columns=data_colums)

        df.iloc[: self.config.horizon + 1, :] = 0

        df["date"] = date
        df = df.set_index("date")
        new_df = df.iloc[1:]
        test = pd.concat([test, new_df])
        return test

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name
    
    def forecast_fit(
        self, train_valid_data: pd.DataFrame, train_ratio_in_tv: float
    ) -> "ModelBase":
        """
        Train the model.

        :param train_data: Time series data used for training.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """

        if train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(train_valid_data)
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)
            
        config = self.config
        # import pdb
        # pdb.set_trace()
        self.model = UniversalRWKVTimeSeries(self.config)
        if config.precision == 'bf16':
            self.model =self.model.to(dtype=torch.bfloat16)
        print(
            "----------------------------------------------------------",
            self.model_name,
        )

        
        

        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, config.seq_len
        ) # seq_len 重叠长度

        self.scaler.fit(train_data.values)

        if config.norm:
            train_data = pd.DataFrame(
                self.scaler.transform(train_data.values),
                columns=train_data.columns,
                index=train_data.index,
            )

        if train_ratio_in_tv != 1:
            if config.norm:
                valid_data = pd.DataFrame(
                    self.scaler.transform(valid_data.values),
                    columns=valid_data.columns,
                    index=valid_data.index,
                )
            valid_dataset, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
                
            )

        train_dataset, train_data_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
        )
        # import pdb
        # pdb.set_trace()
        # Define the loss function and optimizer
        if config.loss == "MSE":
            criterion = nn.MSELoss()
        elif config.loss == "MAE":
            criterion = nn.L1Loss()
        else:
            criterion = nn.HuberLoss(delta=0.5)

        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stopping = EarlyStopping(patience=config.patience)
        self.model.to(device)

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Total trainable parameters: {total_params}")

        for epoch in range(config.num_epochs):
            self.model.train()
            # for input, target, input_mark, target_mark in train_data_loader:
            for i, (input, target, input_mark, target_mark) in enumerate(
                    train_data_loader
            ):
                optimizer.zero_grad()
                input, _target, _input_mark, _target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                # decoder input

                output = self.model(input)

                target = input[:, config.horizon:, :]
                output = output[:, :-config.horizon, :]
                loss = criterion(output, target)

                total_loss = loss
                total_loss.backward()

                optimizer.step()

            if train_ratio_in_tv != 1:
                valid_loss = self.validate(valid_data_loader, criterion)
                self.early_stopping(valid_loss, self.model)
                if self.early_stopping.early_stop:
                    break

            adjust_learning_rate(optimizer, epoch + 1, config)

    def forecast(self, horizon: int, train: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        :param horizon: The predicted length.
        :param testdata: Time data data used for prediction.
        :return: An array of predicted results.
        """
        if self.early_stopping.check_point is not None:
            self.model.load_state_dict(self.early_stopping.check_point)

        if self.config.norm:
            train = pd.DataFrame(
                self.scaler.transform(train.values),
                columns=train.columns,
                index=train.index,
            )

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config
        train, test = split_before(train, len(train) - config.seq_len)

        # Additional timestamp marks required to generate transformer class methods
        test = self.padding_data_for_forecast(test)

        test_data_set, test_data_loader = forecasting_data_provider(
            test, config, timeenc=1, batch_size=1, shuffle=False, drop_last=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            answer = None
            while answer is None or answer.shape[0] < horizon:
                for input, target, input_mark, target_mark in test_data_loader:
                    input, target, input_mark, target_mark = (
                        input.to(device),
                        target.to(device),
                        input_mark.to(device),
                        target_mark.to(device),
                    )

                    output = self.model(input)

                column_num = output.shape[-1]
                temp = output.cpu().float().numpy().reshape(-1, column_num)[-config.horizon:]

                if answer is None:
                    answer = temp
                else:
                    answer = np.concatenate([answer, temp], axis=0)

                if answer.shape[0] >= horizon:
                    if self.config.norm:
                        answer[-horizon:] = self.scaler.inverse_transform(
                            answer[-horizon:]
                        )
                    return answer[-horizon:]

                output = output.cpu().numpy()[:, -config.horizon:, :]
                for i in range(config.horizon):
                    test.iloc[i + config.seq_len] = output[0, i, :]

                test = test.iloc[config.horizon:]
                test = self.padding_data_for_forecast(test)

                test_data_set, test_data_loader = forecasting_data_provider(
                    test,
                    config,
                    timeenc=1,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                )
                
    def validate(self, valid_data_loader, criterion):
        config = self.config
        total_loss = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for input, _target, _input_mark, _target_mark in valid_data_loader:
                input, target, input_mark, target_mark = (
                    input.to(device),
                    _target.to(device),
                    _input_mark.to(device),
                    _target_mark.to(device),
                )

                output = self.model(input)
                target = input[:, config.horizon:, :].contiguous()
                output = output[:, :-config.horizon, :].contiguous()
                loss = criterion(output, target).detach().cpu().float().numpy()
                total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss