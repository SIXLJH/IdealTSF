import os
import copy
import torch
import random
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

try:
    from .utils.attention import scaled_dot_product_attention
    from .utils.dataset import LabeledDataset
    from .utils.revin import RevIN
    from .utils.ECOS import ECOS
    from .utils.channel_bridge import LightChannelBridge
    from .utils.training_data import (as_dataset, infer_dimensions, evaluate_streaming,
                                      supervised_loss, cpu_state_dict)
except ImportError:
    from utils.attention import scaled_dot_product_attention
    from utils.dataset import LabeledDataset
    from utils.revin import RevIN
    from utils.ECOS import ECOS
    from utils.channel_bridge import LightChannelBridge
    from utils.training_data import (as_dataset, infer_dimensions, evaluate_streaming,
                                      supervised_loss, cpu_state_dict)


class IdealTSFArchitecture(nn.Module):
    def __init__(
        self,
        num_channels,
        seq_len,
        hid_dim,
        pred_horizon,
        use_revin=True,
        ca_mode="off",
        num_bridges=16,
        **unused_kwargs
    ):
        super().__init__()

        self.revin = RevIN(num_features=num_channels, condition_dim=20)

        self.compute_keys = nn.Linear(seq_len, hid_dim)
        self.compute_queries = nn.Linear(seq_len, hid_dim)
        self.compute_values = nn.Linear(seq_len, seq_len)

        self.linear_forecaster = nn.Linear(seq_len, pred_horizon)

        self.use_revin = use_revin
        if ca_mode not in {"off", "bridge"}:
            raise ValueError("ca_mode must be off or bridge.")
        self.ca_mode = ca_mode
        self.channel_bridge = LightChannelBridge(hid_dim, num_bridges) if ca_mode == "bridge" else None
        self.model_config = dict(num_channels=num_channels, seq_len=seq_len,
                                 hid_dim=hid_dim, pred_horizon=pred_horizon,
                                 use_revin=use_revin, ca_mode=ca_mode, num_bridges=num_bridges)

    def forward(self, x, flatten_output=True):
        if self.use_revin:
            x_norm = self.revin(x.transpose(1, 2), mode="norm").transpose(1, 2)
        else:
            x_norm = x

        queries = self.compute_queries(x_norm)
        keys = self.compute_keys(x_norm)
        values = self.compute_values(x_norm)

        if self.channel_bridge is not None:
            att_score = self.channel_bridge(queries, keys, values)
        elif hasattr(nn.functional, "scaled_dot_product_attention"):
            att_score = nn.functional.scaled_dot_product_attention(queries, keys, values, is_causal=False)
        else:
            att_score = scaled_dot_product_attention(queries, keys, values, is_causal=False)

        attention_weights = att_score

        out = x_norm + att_score
        out = self.linear_forecaster(out)

        if self.use_revin:
            out = self.revin(out.transpose(1, 2), mode="denorm").transpose(1, 2)

        if flatten_output:
            out = out.reshape([out.shape[0], out.shape[1] * out.shape[2]])

        return out, attention_weights


class TimeSeriesAugmentations:
    def __init__(
        self,
        noise_std=0.01,
        max_erase_length=300,
        erase_prob=0.1,
        noise_scales=None,
        min_erase_length=4,

        # Safe stable distribution jump noise
        use_stable_jump=False,
        stable_jump_prob=0.05,
        stable_alpha=1.5,
        stable_beta=0.0,
        stable_scale=0.005,
        stable_loc=0.0,
        stable_clip=0.05,
        stable_width=1,

        **unused_kwargs
    ):
        """
        Safe negative-sample augmentation for IdealTSF.

        Main components:
            1. Stable-distribution jump noise: low-probability heavy-tailed local jump.
            2. Multi-scale Gaussian noise.
            3. Structured erase.

        Important:
            Stable jump is disabled by default.
            It should only be used in pretraining, not in fine-tuning/evaluation.
        """
        if noise_scales is None:
            noise_scales = [0.1, 0.05, 0.02]

        self.noise_std = noise_std
        self.erase_prob = erase_prob
        self.max_erase_length = max_erase_length
        self.noise_scales = noise_scales
        self.min_erase_length = min_erase_length

        self.use_stable_jump = use_stable_jump
        self.stable_jump_prob = stable_jump_prob
        self.stable_alpha = stable_alpha
        self.stable_beta = stable_beta
        self.stable_scale = stable_scale
        self.stable_loc = stable_loc
        self.stable_clip = stable_clip
        self.stable_width = stable_width

    def _sample_symmetric_alpha_stable(self, shape, device, dtype):
        """
        Sample from a symmetric alpha-stable distribution using
        Chambers-Mallows-Stuck sampling.

        For safety:
            - beta is kept as 0 by default.
            - alpha is clipped into [0.8, 1.95].
            - output will be scaled and clipped later.
        """
        alpha = float(self.stable_alpha)
        alpha = max(0.8, min(alpha, 1.95))

        eps = 1e-6

        u = torch.empty(shape, device=device, dtype=dtype).uniform_(
            -np.pi / 2.0 + eps,
            np.pi / 2.0 - eps
        )

        w = torch.empty(shape, device=device, dtype=dtype).exponential_(1.0)
        w = torch.clamp(w, min=eps)

        if abs(alpha - 1.0) > 1e-3:
            part1 = torch.sin(alpha * u) / torch.pow(
                torch.clamp(torch.cos(u), min=eps),
                1.0 / alpha
            )

            part2 = torch.pow(
                torch.clamp(torch.cos((1.0 - alpha) * u), min=eps) / w,
                (1.0 - alpha) / alpha
            )

            sample = part1 * part2
        else:
            # Cauchy-like special case.
            sample = torch.tan(u)

        return sample

    def stable_jump_noise(self, x):
        """
        Add safe heavy-tailed local jumps.

        x: (B, C, L)

        Design:
            - Apply to only part of samples.
            - Add jump at random local positions.
            - Use very small scale.
            - Clip the perturbation amplitude.
        """
        if not self.use_stable_jump:
            return x

        if self.stable_jump_prob <= 0 or self.stable_scale <= 0:
            return x

        x_aug = x.clone()

        device = x_aug.device
        dtype = x_aug.dtype

        batch_size, num_channels, seq_len = x_aug.shape

        jump_width = int(self.stable_width)
        jump_width = max(1, min(jump_width, seq_len))

        sample_mask = torch.rand(batch_size, device=device) < self.stable_jump_prob

        if sample_mask.sum().item() == 0:
            return x_aug

        # One stable jump value for each sample and channel.
        stable_noise = self._sample_symmetric_alpha_stable(
            shape=(batch_size, num_channels),
            device=device,
            dtype=dtype
        )

        stable_noise = stable_noise * float(self.stable_scale) + float(self.stable_loc)

        if self.stable_clip is not None and self.stable_clip > 0:
            stable_noise = torch.clamp(
                stable_noise,
                min=-float(self.stable_clip),
                max=float(self.stable_clip)
            )

        if seq_len - jump_width + 1 <= 1:
            start_times = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            start_times = torch.randint(
                low=0,
                high=seq_len - jump_width + 1,
                size=(batch_size,),
                device=device
            )

        for i in range(batch_size):
            if sample_mask[i]:
                start_idx = int(start_times[i].item())
                end_idx = start_idx + jump_width

                x_aug[i, :, start_idx:end_idx] = (
                    x_aug[i, :, start_idx:end_idx]
                    + stable_noise[i].unsqueeze(-1)
                )

        return x_aug

    def add_multiscale_noise(self, x):
        device = x.device
        n_samples, n_channels, n_timesteps = x.size()

        noise = torch.zeros_like(x, device=device)

        for i, scale in enumerate(self.noise_scales):
            noise_level = scale * self.noise_std
            noise_component = torch.randn_like(x) * noise_level

            window_size = 2 ** (i + 1)

            if window_size >= n_timesteps:
                noise_component = noise_component.mean(dim=2, keepdim=True)
                noise_component = noise_component.repeat(1, 1, n_timesteps)
            else:
                noise_component = noise_component.unfold(2, window_size, 1).mean(dim=3)

                if noise_component.size(2) < n_timesteps:
                    padding_size = n_timesteps - noise_component.size(2)
                    noise_component = torch.cat(
                        [
                            noise_component,
                            torch.zeros(
                                n_samples,
                                n_channels,
                                padding_size,
                                device=device
                            )
                        ],
                        dim=2
                    )
                elif noise_component.size(2) > n_timesteps:
                    noise_component = noise_component[:, :, :n_timesteps]

            noise = noise + noise_component

        return x + noise

    def structured_erase(self, x):
        if np.random.rand() < self.erase_prob:
            max_length = min(self.max_erase_length, x.size(2) // 4)

            if max_length >= self.min_erase_length + 1:
                erase_length = np.random.randint(
                    self.min_erase_length,
                    max_length + 1
                )

                start_idx = np.random.randint(0, x.size(2) - erase_length)
                x[:, :, start_idx:start_idx + erase_length] = 0

        return x

    def augment(self, x, use_stable_jump=None):
        """
        use_stable_jump:
            None: use self.use_stable_jump
            True: force enable stable jump
            False: force disable stable jump
        """
        if use_stable_jump is None:
            enable_stable = self.use_stable_jump
        else:
            enable_stable = bool(use_stable_jump)

        if enable_stable:
            x = self.stable_jump_noise(x)

        x = self.add_multiscale_noise(x)
        x = self.structured_erase(x)

        return x

    @staticmethod
    def preprocess_series(x, z_thresh=2.5, use_iqr=True, window_size=5):
        assert x.ndim == 3, "Input tensor must be 3D: (batch, channels, timesteps)"

        device = x.device

        nan_mask = torch.isnan(x)

        def linear_interpolate(seq):
            left = seq.clone()
            right = seq.clone()

            for t in range(1, seq.size(-1)):
                left[:, :, t] = torch.where(
                    torch.isnan(left[:, :, t]),
                    left[:, :, t - 1],
                    left[:, :, t]
                )

            for t in reversed(range(seq.size(-1) - 1)):
                right[:, :, t] = torch.where(
                    torch.isnan(right[:, :, t]),
                    right[:, :, t + 1],
                    right[:, :, t]
                )

            interp = 0.5 * (left + right)
            return interp

        x = torch.where(nan_mask, torch.tensor(0.0, device=device), x)
        x = linear_interpolate(x)

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6

        z_score = (x - mean) / std
        outlier_mask = z_score.abs() > z_thresh

        if use_iqr:
            q1 = x.quantile(0.25, dim=-1, keepdim=True)
            q3 = x.quantile(0.75, dim=-1, keepdim=True)
            iqr = q3 - q1

            iqr_mask = (x < (q1 - 1.5 * iqr)) | (x > (q3 + 1.5 * iqr))
            outlier_mask = outlier_mask | iqr_mask

        kernel = torch.ones(1, 1, 3, device=device) / 3.0
        x_padded = F.pad(x, (1, 1), mode="replicate")

        local_mean = F.conv1d(
            x_padded.view(-1, 1, x.size(-1) + 2),
            kernel
        ).view(x.shape)

        x = torch.where(outlier_mask, local_mean, x)

        avg_kernel = torch.ones(1, 1, window_size, device=device) / window_size
        pad = window_size // 2

        x_padded = F.pad(x, (pad, pad), mode="reflect")

        x_smooth = F.conv1d(
            x_padded.view(-1, 1, x.size(-1) + 2 * pad),
            avg_kernel
        ).view(x.shape)

        return x_smooth


class IdealTSF:
    def __init__(
        self,
        device="cuda:0",
        num_epochs=100,
        batch_size=256,
        base_optimizer=torch.optim.AdamW,
        learning_rate=1e-3,
        weight_decay=1e-5,
        rho=0.5,
        use_revin=True,
        random_state=None,
        use_preprocessing=False,
        augmentations=None,
        ca_mode="off",
        num_bridges=16,
        hid_dim=16,
        **unused_kwargs
    ):
        self.network = None
        self.criterion = nn.L1Loss()
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.base_optimizer = base_optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rho = rho
        self.use_revin = use_revin
        self.random_state = random_state
        self.augmentations = augmentations
        self.use_preprocessing = use_preprocessing
        self.ca_mode, self.num_bridges = ca_mode, num_bridges
        self.hid_dim = hid_dim

    def fit(self, x, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)
            np.random.seed(self.random_state)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        self.network = IdealTSFArchitecture(
            num_channels=x.shape[1],
            seq_len=x.shape[2],
            hid_dim=self.hid_dim,
            pred_horizon=y.shape[1] // x.shape[1],
            use_revin=self.use_revin, ca_mode=self.ca_mode, num_bridges=self.num_bridges
        )

        self.criterion = self.criterion.to(self.device)
        self.network = self.network.to(self.device)
        self.network.train()

        optimizer = ECOS(
            self.network.parameters(),
            base_optimizer=self.base_optimizer,
            rho=self.rho,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        train_dataset = LabeledDataset(x, y)

        data_loader_train = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        progress_bar = tqdm(range(self.num_epochs))

        for epoch in progress_bar:
            loss_list = []

            for x_batch, y_batch in data_loader_train:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                if self.augmentations:
                    x_batch = self.augmentations.augment(x_batch)

                out_batch, _ = self.network(x_batch)
                loss = self.criterion(out_batch, y_batch)

                if isinstance(optimizer, ECOS):
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    out_batch, _ = self.network(x_batch)
                    loss = self.criterion(out_batch, y_batch)

                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_list.append(loss.item())

            train_loss = np.mean(loss_list)

            progress_bar.set_description(
                f"Epoch {epoch}: Train Loss {train_loss:.4f}",
                refresh=True
            )

        return

    def forecast(self, x, batch_size=256):
        self.network.eval()

        outs = []
        n = x.shape[0]

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)

                if isinstance(x, torch.Tensor):
                    xb = x[start:end].detach().float().to(self.device)
                else:
                    xb = torch.from_numpy(x[start:end]).float().to(self.device)

                out, _ = self.network(xb)
                outs.append(out.detach().cpu())

                del xb, out

        outs = torch.cat(outs, dim=0).numpy()

        self.network.train()

        return outs

    def predict(self, x, batch_size=256):
        return self.forecast(x, batch_size=batch_size)


class Pretrainer:
    def __init__(
        self,
        device="cuda:0",
        num_epochs=50,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-5,
        rho=0.5,
        augmentations=None,
        use_revin=True,
        ca_mode="off",
        num_bridges=16,
        num_workers=0,
        hid_dim=16,
        **unused_kwargs
    ):
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rho = rho
        self.use_revin = use_revin
        self.augmentations = augmentations
        self.network = None
        self.ca_mode, self.num_bridges = ca_mode, num_bridges
        self.num_workers = num_workers
        self.hid_dim = hid_dim
        self.history = []

    @torch.no_grad()
    def _evaluate(self, x, y=None, eval_batch_size=None):
        return evaluate_streaming(self.network, x, y, device=self.device,
                                  batch_size=eval_batch_size or self.batch_size)

    def pretrain(
        self,
        x,
        y=None,
        x_val=None,
        y_val=None,
        x_test=None,
        y_test=None,
        eval_batch_size=None
    ):
        channels, seq_len, pred_len = infer_dimensions(x, y)
        self.network = IdealTSFArchitecture(
            num_channels=channels, seq_len=seq_len, hid_dim=self.hid_dim,
            pred_horizon=pred_len, use_revin=self.use_revin,
            ca_mode=self.ca_mode, num_bridges=self.num_bridges,
        ).to(self.device)

        optimizer = ECOS(
            self.network.parameters(),
            base_optimizer=torch.optim.AdamW,
            rho=self.rho,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        criterion = nn.L1Loss().to(self.device)

        dataset = as_dataset(x, y)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.num_workers, drop_last=False)
        self.target_index = getattr(dataset, "target_index", None)

        self.network.train()

        for epoch in tqdm(range(self.num_epochs), desc="[Pretraining]"):
            epoch_losses = []

            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                if self.augmentations:
                    xb = self.augmentations.augment(xb)

                out, _ = self.network(xb)
                loss = supervised_loss(self.network, out, yb, criterion, self.target_index)

                if isinstance(optimizer, ECOS):
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    out, _ = self.network(xb)
                    loss = supervised_loss(self.network, out, yb, criterion, self.target_index)

                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                epoch_losses.append(loss.item())

            train_loss = float(np.mean(epoch_losses))

            msg = (
                f"[Pretrain Epoch {epoch + 1:03d}/{self.num_epochs:03d}] "
                f"Train Loss: {train_loss:.6f}"
            )

            if x_val is not None and (y_val is not None or isinstance(x_val, Dataset)):
                val_mse, val_mae = self._evaluate(
                    x_val,
                    y_val,
                    eval_batch_size=eval_batch_size
                )
                msg += f" | Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f}"

            if x_test is not None and (y_test is not None or isinstance(x_test, Dataset)):
                test_mse, test_mae = self._evaluate(
                    x_test,
                    y_test,
                    eval_batch_size=eval_batch_size
                )
                msg += f" | Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f}"

            self.history.append({"epoch": epoch + 1, "train_loss": train_loss,
                                 "val_mse": locals().get("val_mse"),
                                 "val_mae": locals().get("val_mae")})
            print(msg)

        return self.network


class Trainer:
    def __init__(
        self,
        pretrained_model,
        device="cuda:0",
        num_epochs=50,
        batch_size=256,
        learning_rate=1e-4,
        weight_decay=1e-5,
        rho=0.5,
        use_revin=True,

        # Safe positive-sample auxiliary training
        use_positive_aux=False,
        positive_loss_weight=0.2,
        positive_window_size=5,
        positive_mix_alpha=0.85,
        positive_use_preprocess=False,
        positive_z_thresh=2.5,
        positive_use_iqr=True,
        num_workers=0,
        patience=0,

        **unused_kwargs
    ):
        self.network = pretrained_model
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rho = rho
        self.use_revin = use_revin

        # Positive sample auxiliary settings
        self.use_positive_aux = use_positive_aux
        self.positive_loss_weight = positive_loss_weight
        self.positive_window_size = positive_window_size
        self.positive_mix_alpha = positive_mix_alpha
        self.positive_use_preprocess = positive_use_preprocess
        self.positive_z_thresh = positive_z_thresh
        self.positive_use_iqr = positive_use_iqr

        self.num_workers, self.patience = num_workers, patience
        self.history = []
        self.target_index = None
        self.best_val_mse = None
        self.best_val_mae = None
        self.best_test_mse = None
        self.best_test_mae = None
        self.best_epoch = None
        self.best_state_dict = None

    @torch.no_grad()
    def _evaluate(self, x, y=None, eval_batch_size=None):
        return evaluate_streaming(self.network, x, y, device=self.device,
                                  batch_size=eval_batch_size or self.batch_size)

    def _moving_average_positive_sample(self, x):
        """
        Generate a smoothed positive sample.

        x: (B, C, L)

        Blend a lightweight moving average with the original input:
            x_pos = alpha * x + (1 - alpha) * x_smooth

        Larger alpha retains more of the original sample.
        """
        window_size = int(self.positive_window_size)

        if window_size <= 1:
            return x

        # Use an odd window to preserve alignment.
        if window_size % 2 == 0:
            window_size += 1

        pad = window_size // 2

        x_pad = F.pad(x, (pad, pad), mode="replicate")

        x_smooth = F.avg_pool1d(
            x_pad,
            kernel_size=window_size,
            stride=1,
        )

        # Preserve the input length across PyTorch versions.
        if x_smooth.shape[-1] > x.shape[-1]:
            x_smooth = x_smooth[:, :, :x.shape[-1]]
        elif x_smooth.shape[-1] < x.shape[-1]:
            x_smooth = F.pad(
                x_smooth,
                (0, x.shape[-1] - x_smooth.shape[-1]),
                mode="replicate",
            )

        alpha = float(self.positive_mix_alpha)
        alpha = max(0.0, min(1.0, alpha))

        x_pos = alpha * x + (1.0 - alpha) * x_smooth

        return x_pos

    def _make_positive_sample(self, x):
        """
        Generate positive samples.

        Use a moving-average positive sample by default.
        When positive_use_preprocess is enabled, apply missing-value handling,
        outlier replacement and smoothing before blending with the input.
        """
        if self.positive_use_preprocess:
            x_clean = TimeSeriesAugmentations.preprocess_series(
                x,
                z_thresh=self.positive_z_thresh,
                use_iqr=self.positive_use_iqr,
                window_size=self.positive_window_size,
            )

            alpha = float(self.positive_mix_alpha)
            alpha = max(0.0, min(1.0, alpha))

            x_pos = alpha * x + (1.0 - alpha) * x_clean

            return x_pos

        return self._moving_average_positive_sample(x)

    def _compute_train_loss(self, xb, yb, criterion):
        """
        Main loss:
            L_orig = L(f(x), y)

        Positive auxiliary loss:
            L_pos = L(f(x_pos), y)

        Total:
            L = L_orig + lambda_pos * L_pos

        Positive augmentation affects training only.
        Inference uses the original input x.
        """
        out, _ = self.network(xb)
        loss_orig = supervised_loss(self.network, out, yb, criterion, self.target_index)

        if self.use_positive_aux and self.positive_loss_weight > 0:
            xb_pos = self._make_positive_sample(xb)
            out_pos, _ = self.network(xb_pos)

            loss_pos = supervised_loss(self.network, out_pos, yb, criterion, self.target_index)

            loss = loss_orig + float(self.positive_loss_weight) * loss_pos
        else:
            loss_pos = None
            loss = loss_orig

        return loss, loss_orig, loss_pos

    def fine_tune(
        self,
        x,
        y=None,
        x_val=None,
        y_val=None,
        x_test=None,
        y_test=None,
        eval_batch_size=None,
        restore_best=True,
        save_best_path=None
    ):
        """
        Fine-tune IdealTSF with optional positive auxiliary loss.

        Select the best model using validation MSE only.

        If provided explicitly, test data is used for diagnostic logging only
        and never participates in model selection.
        """

        optimizer = ECOS(
            self.network.parameters(),
            base_optimizer=torch.optim.AdamW,
            rho=self.rho,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        criterion = nn.L1Loss().to(self.device)

        dataset = as_dataset(x, y)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.num_workers, drop_last=False)
        self.target_index = getattr(dataset, "target_index", None)

        self.network.to(self.device)
        self.network.train()

        has_val = x_val is not None and (y_val is not None or isinstance(x_val, Dataset))
        has_test = x_test is not None and (y_test is not None or isinstance(x_test, Dataset))

        best_val_mse = float("inf")
        best_val_mae = float("inf")
        best_test_mse = None
        best_test_mae = None
        best_epoch = -1
        best_state_dict = cpu_state_dict(self.network)
        epochs_without_improvement = 0

        for epoch in tqdm(range(self.num_epochs), desc="[Fine-tuning]"):
            epoch_losses = []
            epoch_orig_losses = []
            epoch_pos_losses = []

            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                if isinstance(optimizer, ECOS):
                    loss, loss_orig, loss_pos = self._compute_train_loss(
                        xb=xb,
                        yb=yb,
                        criterion=criterion
                    )

                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    loss, loss_orig, loss_pos = self._compute_train_loss(
                        xb=xb,
                        yb=yb,
                        criterion=criterion
                    )

                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.zero_grad()

                    loss, loss_orig, loss_pos = self._compute_train_loss(
                        xb=xb,
                        yb=yb,
                        criterion=criterion
                    )

                    loss.backward()
                    optimizer.step()

                epoch_losses.append(loss.item())
                epoch_orig_losses.append(loss_orig.item())

                if loss_pos is not None:
                    epoch_pos_losses.append(loss_pos.item())

            train_loss = float(np.mean(epoch_losses))
            train_orig_loss = float(np.mean(epoch_orig_losses))

            if len(epoch_pos_losses) > 0:
                train_pos_loss = float(np.mean(epoch_pos_losses))
            else:
                train_pos_loss = None

            msg = (
                f"[Fine-tune Epoch {epoch + 1:03d}/{self.num_epochs:03d}] "
                f"Train Loss: {train_loss:.6f} | "
                f"Orig Loss: {train_orig_loss:.6f}"
            )

            if train_pos_loss is not None:
                msg += (
                    f" | Pos Loss: {train_pos_loss:.6f}"
                    f" | Pos Weight: {self.positive_loss_weight:.4f}"
                )

            current_val_mse = None
            current_val_mae = None
            current_test_mse = None
            current_test_mae = None

            if has_val:
                current_val_mse, current_val_mae = self._evaluate(
                    x_val,
                    y_val,
                    eval_batch_size=eval_batch_size
                )

                msg += (
                    f" | Val MSE: {current_val_mse:.6f}"
                    f" | Val MAE: {current_val_mae:.6f}"
                )

            if has_test:
                current_test_mse, current_test_mae = self._evaluate(
                    x_test,
                    y_test,
                    eval_batch_size=eval_batch_size
                )

                msg += (
                    f" | Test MSE: {current_test_mse:.6f}"
                    f" | Test MAE: {current_test_mae:.6f}"
                )

            if has_val and current_val_mse < best_val_mse:
                best_val_mse = current_val_mse
                best_val_mae = current_val_mae
                best_test_mse = current_test_mse
                best_test_mae = current_test_mae
                best_epoch = epoch + 1
                best_state_dict = cpu_state_dict(self.network)
                epochs_without_improvement = 0

                if save_best_path is not None:
                    save_dir = os.path.dirname(save_best_path)

                    if save_dir != "":
                        os.makedirs(save_dir, exist_ok=True)

                    torch.save(
                        {
                            "model_state_dict": best_state_dict,
                            "model_config": self.network.model_config,
                            "best_epoch": best_epoch,
                            "best_val_mse": best_val_mse,
                            "best_val_mae": best_val_mae,
                            "corresponding_test_mse": best_test_mse,
                            "corresponding_test_mae": best_test_mae,
                            "use_positive_aux": self.use_positive_aux,
                            "positive_loss_weight": self.positive_loss_weight,
                            "positive_window_size": self.positive_window_size,
                            "positive_mix_alpha": self.positive_mix_alpha,
                            "positive_use_preprocess": self.positive_use_preprocess,
                        },
                        save_best_path
                    )

                msg += (
                    f" | *Best Val Updated*"
                    f" | Best Val MSE: {best_val_mse:.6f}"
                    f" | Best Epoch: {best_epoch}"
                )

                if best_test_mse is not None:
                    msg += (
                        f" | Corresponding Test MSE: {best_test_mse:.6f}"
                        f" | Corresponding Test MAE: {best_test_mae:.6f}"
                    )

            elif has_val:
                epochs_without_improvement += 1
                msg += (
                    f" | Best Val MSE: {best_val_mse:.6f}"
                    f" | Best Epoch: {best_epoch}"
                )

                if best_test_mse is not None:
                    msg += (
                        f" | Corresponding Test MSE: {best_test_mse:.6f}"
                        f" | Corresponding Test MAE: {best_test_mae:.6f}"
                    )

            self.history.append({"epoch": epoch + 1, "train_loss": train_loss,
                                 "original_loss": train_orig_loss, "positive_loss": train_pos_loss,
                                 "val_mse": current_val_mse, "val_mae": current_val_mae,
                                 "is_best": best_epoch == epoch + 1})
            print(msg)
            if has_val and self.patience > 0 and epochs_without_improvement >= self.patience:
                print(f"[Early stopping] no improvement for {self.patience} epochs")
                break

        self.best_val_mse = best_val_mse if has_val else None
        self.best_val_mae = best_val_mae if has_val else None
        self.best_test_mse = best_test_mse
        self.best_test_mae = best_test_mae
        self.best_epoch = best_epoch if has_val else None
        self.best_state_dict = best_state_dict if has_val else None

        if restore_best and has_val:
            self.network.load_state_dict(best_state_dict)

            print(
                f"[Restore Best-Val Model] "
                f"Best Epoch: {best_epoch}, "
                f"Best Val MSE: {best_val_mse:.6f}, "
                f"Best Val MAE: {best_val_mae:.6f}"
            )

            if best_test_mse is not None:
                print(
                    f"[Best-Val Corresponding Test] "
                    f"Test MSE: {best_test_mse:.6f}, "
                    f"Test MAE: {best_test_mae:.6f}"
                )

        elif restore_best and not has_val:
            print(
                "[Warning] x_val/y_val were not provided, "
                "so the model was not selected by validation set."
            )

    def forecast(self, x, eval_batch_size=None, return_attention=False):
        self.network.eval()

        if eval_batch_size is None:
            eval_batch_size = self.batch_size

        preds = []
        attn_weights = [] if return_attention else None

        n = x.shape[0]

        with torch.no_grad():
            for start in range(0, n, eval_batch_size):
                end = min(start + eval_batch_size, n)

                if isinstance(x, torch.Tensor):
                    xb = x[start:end].detach().float().to(self.device)
                else:
                    xb = torch.from_numpy(x[start:end]).float().to(self.device)

                out, attn = self.network(xb)

                preds.append(out.detach().cpu())

                if return_attention:
                    attn_weights.append(attn.detach().cpu())

                del xb, out, attn

        preds = torch.cat(preds, dim=0).numpy()

        if return_attention:
            try:
                attn_weights = torch.cat(attn_weights, dim=0).numpy()
            except Exception:
                attn_weights = None
        else:
            attn_weights = None

        self.network.train()

        return preds, attn_weights

    def predict(self, x, eval_batch_size=None):
        pred, _ = self.forecast(
            x,
            eval_batch_size=eval_batch_size,
            return_attention=False
        )
        return pred
