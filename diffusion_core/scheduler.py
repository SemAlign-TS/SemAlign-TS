import math
import numpy as np
import torch
import torch.nn.functional as F


class GaussianDiffusion:
    def __init__(
        self,
        model,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        device="cuda",
        schedule="linear",
    ):
        self.model = model.to(device)
        self.timesteps = int(timesteps)
        self.device = torch.device(device)
        self.schedule = schedule

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.timesteps)

        elif schedule == "cosine":
            betas = self._cosine_beta_schedule(self.timesteps)

        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

        self.betas = betas.to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(
            ((x / timesteps) + s) / (1 + s) * math.pi * 0.5
        ) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min=1e-8, max=0.999)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def p_losses(self, x_0, t, text_emb, text_mask=None, noise=None, reduction="mean"):
        if noise is None:
            noise = torch.randn_like(x_0)

        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t, text_emb, text_mask)

        loss = F.mse_loss(predicted_noise, noise, reduction="none")

        if reduction == "none":
            return loss.mean(dim=list(range(1, len(loss.shape))))
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()

        raise ValueError(f"Unknown reduction: {reduction}")

    @torch.no_grad()
    def sample(self, text_emb, text_mask, shape):
        training_before = self.model.training
        self.model.eval()

        b = shape[0]
        img = torch.randn(shape, device=self.device)

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)

            noise_pred = self.model(img, t, text_emb, text_mask)

            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_bar = self.alphas_cumprod[i]

            mean = (1.0 / torch.sqrt(alpha)) * (
                img - (beta / torch.sqrt(1.0 - alpha_bar)) * noise_pred
            )

            if i > 0:
                noise = torch.randn_like(img)
                sigma = torch.sqrt(beta)
                img = mean + sigma * noise
            else:
                img = mean

        self.model.train(training_before)
        return img

    @torch.no_grad()
    def ddim_sample(
        self,
        text_emb,
        text_mask,
        shape,
        ddim_steps=50,
        eta=0.0,
        cfg_scale=1.0,
        init_noise=None,
    ):
        """
        DDIM sampling for time-series diffusion.

        The model was not trained with classifier-free condition dropout,
        so cfg_scale must remain 1.0.
        """
        if cfg_scale != 1.0:
            raise NotImplementedError(
                "CFG is not supported because the model was not trained with "
                "condition dropout. Please use cfg_scale=1.0."
            )

        training_before = self.model.training
        self.model.eval()

        b = shape[0]

        if init_noise is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = init_noise.to(self.device)

        time_steps = np.linspace(1, self.timesteps - 1, ddim_steps, dtype=int)

        for i in reversed(range(len(time_steps))):
            t_cur = int(time_steps[i])
            t_prev = int(time_steps[i - 1]) if i > 0 else 0

            t_tensor = torch.full((b,), t_cur, device=self.device, dtype=torch.long)

            noise_pred = self.model(img, t_tensor, text_emb, text_mask)

            alpha_bar_cur = self.alphas_cumprod[t_cur]
            alpha_bar_prev = (
                self.alphas_cumprod[t_prev]
                if i > 0
                else torch.tensor(1.0, device=self.device)
            )

            pred_x0 = (
                img
                - torch.sqrt(torch.clamp(1.0 - alpha_bar_cur, min=0.0)) * noise_pred
            ) / torch.sqrt(torch.clamp(alpha_bar_cur, min=1e-8))

            sigma = eta * torch.sqrt(
                torch.clamp(
                    (1.0 - alpha_bar_prev)
                    / torch.clamp(1.0 - alpha_bar_cur, min=1e-8)
                    * (
                        1.0
                        - alpha_bar_cur / torch.clamp(alpha_bar_prev, min=1e-8)
                    ),
                    min=0.0,
                )
            )

            dir_coef = 1.0 - alpha_bar_prev - sigma ** 2
            dir_coef = torch.clamp(dir_coef, min=0.0)

            dir_xt = torch.sqrt(dir_coef) * noise_pred

            img = torch.sqrt(torch.clamp(alpha_bar_prev, min=0.0)) * pred_x0 + dir_xt

            if eta > 0 and i > 0:
                img = img + sigma * torch.randn_like(img)

            if not torch.isfinite(img).all():
                raise FloatingPointError(
                    f"Non-finite sample detected at DDIM step {t_cur}"
                )

        self.model.train(training_before)
        return img