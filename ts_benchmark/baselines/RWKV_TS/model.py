########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.strategies import DeepSpeedStrategy

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import pandas as pd
from .utils import compress_parameter_names
import matplotlib.pyplot as plt
def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop

# if os.environ["RWKV_JIT_ON"] == "1":
MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method

# HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
HEAD_SIZE = 64
CHUNK_LEN = 16
########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
load(name="wind_backstepping", sources=[f'ts_benchmark/baselines/RWKV_TS/cuda/wkv7_cuda.cu', 'ts_benchmark/baselines/RWKV_TS/cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db

def RWKV7_OP(q,w,k,v,a,b):
    B,T,HC = q.shape
    q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)
    

########################################################################################################
# RWKV TimeMix
########################################################################################################

class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            # D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            if self.layer_id != 0: # not needed for the first layer
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2)) # !!! notice eps value !!!

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()


    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RWKV7_OP(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first

########################################################################################################
# RWKV ChannelMix
########################################################################################################
class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()

    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)
    
########################################################################################################
# RWKV Block
########################################################################################################

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd) # only used in block 0, should be fused with emb
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class WaveNetEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, kernel_size=2, dilation_base=2):
        super(WaveNetEmbedding, self).__init__()
        self.layers = nn.ModuleList()
        current_dilation = 1
        for _ in range(num_layers):
            current_padding = (kernel_size - 1) * current_dilation
            # 1D 因果卷积层
            self.layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,  # 因果卷积通常使用 kernel_size=2
                    dilation=current_dilation,
                    padding=current_padding  # 保持输出长度与输入长度相同
                )
            )
            in_channels = out_channels
            current_dilation *= dilation_base

        self.activation = nn.ReLU()

    def forward(self, x):
        # 输入 x 的形状应为 (batch_size, in_channels, sequence_length)
        for layer in self.layers:
            x = layer(x)
            # 裁剪输出以确保因果性和保持长度
            x = x[:, :, :x.size(2) - (layer.dilation[0])]
            x = self.activation(x)
        # 输出 x 的形状为 (batch_size, out_channels, sequence_length)
        return x

class CausalMovingAverage(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.conv = nn.Conv1d(1, 1, kernel_size=window_size, stride=1, padding=window_size-1, bias=False)
        nn.init.constant_(self.conv.weight, 1/window_size)  # 固定权重
        self.conv.weight.requires_grad_(False)  

    def forward(self, x):
        # [B, T, 1]
        x = x.transpose(1, 2)  # [B, 1, T]
        x = self.conv(x)  # out [B, 1, T + window_size-1]
        x = x[:, :, :-self.window_size+1]  
        return x.transpose(1, 2)  # [B, T, 1]
    
class UniversalRWKVTimeSeries(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = WaveNetEmbedding(1, args.n_embd, args.n_emb_layer, dilation_base=2)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, 1, bias=False)

        self.sma_window = getattr(args, 'sma_window', 3) # default 3

        if self.sma_window > 1:
            self.smooth = CausalMovingAverage(self.sma_window)
        
        # self.best_val_loss = float('inf')

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)


    def configure_optimizers(self):
        zero_weight_decay_group = [p for p in self.parameters() if len(p.squeeze().shape) < 2 and p.requires_grad]
        # add weight decay to len(p.squeeze().shape) >= 2
        weight_decay_group = [p for p in self.parameters() if len(p.squeeze().shape) >= 2 and p.requires_grad] 

        name_of_trainable_params = [n for n, p in self.named_parameters() if p.requires_grad]
        compressed_name_of_trainable_params = compress_parameter_names(name_of_trainable_params)
        rank_zero_info(f"Name of trainable parameters in optimizers: {compressed_name_of_trainable_params}")
        rank_zero_info(f"Number of trainable parameters in optimizers: {len(name_of_trainable_params)}")
        optim_groups = []
        optim_groups = []
        if zero_weight_decay_group:
            optim_groups += [{"params": zero_weight_decay_group, "weight_decay": 0.0}]
        if weight_decay_group:
            if self.args.weight_decay > 0:
                optim_groups += [{"params": weight_decay_group, "weight_decay": self.args.weight_decay}]
                rank_zero_info(f"Number of parameters with weight decay: {len(weight_decay_group)}, with value: {self.args.weight_decay}")
            else:
                optim_groups += [{"params": weight_decay_group, "weight_decay": 0.0}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def pad_left(self, x, num_tokens_to_pad):
        # pad left with eos token embedding
        if num_tokens_to_pad != 0:
            # left padding by add eos token at the beginning
            pad_emb = torch.zeros(
                x.size(0), num_tokens_to_pad, x.size(2),
                device=x.device, dtype=x.dtype
                )
            x = torch.cat((pad_emb, x), dim=1)
        return x

    def unpad(self, x, num_tokens_to_pad):
        # unpad
        if num_tokens_to_pad > 0:
            x = x[:, num_tokens_to_pad:]
        return x

    def forward(self, x):
        args = self.args
        # x: [B, T, 1]
        B, T, C = x.size()
        # reshape x to [B, 1, T]
        x = x.transpose(1, 2)
        x = self.emb(x).transpose(1, 2) # [B, D, T] -> [B, T, D]

        num_tokens_to_pad = (
            CHUNK_LEN - x.size(1) % CHUNK_LEN if x.size(1) % CHUNK_LEN != 0 else 0
        )

        x = self.pad_left(x, num_tokens_to_pad)
        if args.dropout > 0:
            x = self.drop0(x)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            if args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)

        x = self.ln_out(x)
        x = self.head(x)

        if hasattr(self, 'smooth') and self.sma_window > 1:
            x = self.smooth(x)

        return self.unpad(x, num_tokens_to_pad)
    
    
    # def training_step(self, batch, batch_idx):
    #     '''
    #     batch: dict with keys "input_ids", "labels" and "input_text"
    #     '''
    #     seq_x = batch["seq_x"]
    #     if self.args.precision == "bf16":
    #         seq_x = seq_x.bfloat16() 
    #     predicts = self(seq_x)
    #     targets = seq_x
    #     shift_predicts = predicts[..., :-1, :].contiguous() 
    #     shift_targets = targets[..., 1:, :].contiguous()
    #     loss = F.mse_loss(shift_predicts, shift_targets)
    #     return loss

    # def training_step_end(self, batch_parts):
    #     if pl.__version__[0]!='2':
    #         all = self.all_gather(batch_parts)
    #         if self.trainer.is_global_zero:
    #             self.trainer.my_loss_all = all
    
    # def validation_step(self, batch, batch_idx):
    #     seq_x = batch["seq_x"]
    #     if self.args.precision == "bf16":
    #         seq_x = seq_x.bfloat16()
    #     predicts = self(seq_x)
    #     targets = seq_x
        
    #     shift_predicts = predicts[..., :-1, :].contiguous()
    #     shift_targets = targets[..., 1:, :].contiguous()
    #     loss = F.mse_loss(shift_predicts, shift_targets)
        

    #     return {
    #         "loss": loss,
    #         "predicts": shift_predicts.detach(),
    #         "targets": shift_targets.detach()
    #     }

    # def validation_epoch_end(self, outputs):
    #     # 多GPU数据聚合，如果不开多卡可以注释掉这里
    #     all_predicts = self.all_gather(torch.cat([out["predicts"] for out in outputs]))
    #     all_targets = self.all_gather(torch.cat([out["targets"] for out in outputs]))
        
    #     predicts_numpy = all_predicts.cpu().float().numpy()
    #     targets_numpy = all_targets.cpu().float().numpy()
        
    #     save_dir = os.path.join(self.args.proj_dir, "data_records")
    #     os.makedirs(save_dir, exist_ok=True)
    #     self._save_to_excel(predicts_numpy, targets_numpy, save_dir)
        
    #     val_loss = F.mse_loss(all_predicts, all_targets).item()
    #     self.log("val_loss", val_loss, sync_dist=True)
        

    #     if val_loss < self.best_val_loss and self.current_epoch >= 1:
    #         self._save_best_model(val_loss)
    #         self._plot_predictions(outputs, val_loss)

    # # def _save_to_excel(self, preds, targets, save_dir):
    # #     df = pd.DataFrame({
    # #         'predict': preds.flatten(),
    # #         'target': targets.flatten()
    # #     })
    # #     save_path = os.path.join(save_dir, f"epoch_{self.current_epoch}.xlsx")
    # #     df.to_excel(save_path, index=False)

    # def _save_to_excel(self, preds, targets, save_dir):
    #     self._fallback_save(preds, targets, save_dir)


    # def _fallback_save(self, preds, targets, save_dir):
    #     df = pd.DataFrame({
    #         'predict': preds.flatten(),
    #         'target': targets.flatten()
    #     }).iloc[:10000, :]  #太长了会超限
    #     print(df.shape)
        
    #     truncate_path = os.path.join(save_dir, f"epoch_{self.current_epoch}_TRUNCATED.csv")
    #     df.to_csv(truncate_path, index=False)
        
    # def _save_best_model(self, val_loss):
    #     old_model = os.path.join('/home/rwkv/RWKV-TS/Universal-RWKV-TS-main/output_dir', f"best-{self.best_val_loss:.3f}.pth")
    #     if os.path.exists(old_model):
    #         os.remove(old_model)
        
    #     new_model = os.path.join('/home/rwkv/RWKV-TS/Universal-RWKV-TS-main/output_dir', f"best-{val_loss:.3f}.pth")
    #     torch.save(self.state_dict(), new_model)
    #     self.best_val_loss = val_loss

    # def _plot_predictions(self, outputs, val_loss):
    #     plot_dir = os.path.join('/home/rwkv/RWKV-TS/Universal-RWKV-TS-main/output_dir', f"best_plots_{val_loss:.3f}")
    #     os.makedirs(plot_dir, exist_ok=True)
        
    #     time_scales = [
    #         ('day', 1),
    #         ('week', 7),
    #         ('month', 30)
    #     ]
        
    #     for scale_name, window in time_scales:
    #         scale_pred = []
    #         scale_target = []
            
    #         for i in range(0, len(outputs), window):
    #             batch_pred = torch.cat([out["predicts"] for out in outputs[i:i+window]])
    #             batch_target = torch.cat([out["targets"] for out in outputs[i:i+window]])
    #             scale_pred.append(batch_pred.to(torch.float).cpu().numpy())
    #             scale_target.append(batch_target.to(torch.float).cpu().numpy())
            
    #         self._plot_scale(scale_pred, scale_target, plot_dir, scale_name)

    # def _plot_scale(self, preds, targets, save_dir, prefix):
    #     plt.figure(figsize=(15, 6))
    #     for i, (p, t) in enumerate(zip(preds, targets)):
    #         plt.clf()
    #         plt.plot(t.flatten()[:672], label='Target', alpha=0.7)
    #         plt.plot(p.flatten()[:672], label='Prediction', linestyle='--')
    #         plt.title(f"{prefix.capitalize()} {i+1} Prediction vs Target")
    #         plt.legend()
    #         plt.savefig(os.path.join(save_dir, f"{prefix}_{i}.png"), dpi=150)
    #         plt.close()
        
    