import torch
import torch.nn.functional as F


TREND_CENTERS = {
    "strong_up": 0.30,
    "moderate_up": 0.10,
    "stable": 0.00,
    "moderate_down": -0.10,
    "strong_down": -0.30,
}
TREND_SOFT_RADIUS = 0.30


def classify_trend(slope):
    if slope >= 0.15:
        return "strong_up"
    if slope >= 0.05:
        return "moderate_up"
    if slope > -0.05:
        return "stable"
    if slope > -0.15:
        return "moderate_down"
    return "strong_down"


def calc_trend_soft_score(slope_value, target_label, radius=TREND_SOFT_RADIUS):
    center = TREND_CENTERS.get(target_label, 0.0)
    return max(0.0, 1.0 - abs(float(slope_value) - center) / max(radius, 1e-6))


class TimeSeriesReward:
    def __init__(self, device="cuda", reward_config=None, dataset_name="unknown"):
        self.device = device
        self.dataset_name = dataset_name.lower()

        default_config = {
            "use_trend": True,
            "use_volatility": True,
            "use_peak": True,
            "use_mse": True,
            "weight_trend": 2.0,
            "weight_volatility": 1.0,
            "weight_peak": 2.0,
            "weight_mse": 1.5,
            "hit_score": 10.0,
            "mse_temperature": 0.5,
            "volatility_margin_ratio": 0.15,
            "use_relative_reward": True,
            "relative_reward_scale": 5.0,
            "smooth_clip_enabled": True,
            "smooth_clip_scale": 15.0,
            "smooth_clip_max": 30.0,
            "trend_soft_alpha": 0.3,
        }
        self.config = default_config.copy()
        if reward_config:
            self.config.update(reward_config)

        print(f"[Reward] {dataset_name}: semantic + temperature-scaled MSE + smooth compression")

    def extract_features_from_series(self, series):
        batch_size, seq_len, _ = series.shape
        series_2d = series.squeeze(-1)
        features = {}

        x_axis = torch.arange(seq_len, device=self.device).float().unsqueeze(0).expand(batch_size, -1)
        x_mean = x_axis.mean(dim=1, keepdim=True)
        y_mean = series_2d.mean(dim=1, keepdim=True)
        numerator = ((x_axis - x_mean) * (series_2d - y_mean)).sum(dim=1)
        denominator = ((x_axis - x_mean) ** 2).sum(dim=1)
        slope = numerator / (denominator + 1e-8)
        value_range = series_2d.max(dim=1)[0] - series_2d.min(dim=1)[0]
        features["normalized_slope"] = slope * seq_len / (value_range + 1e-8)

        trend_line = slope.unsqueeze(1) * x_axis + (y_mean - slope.unsqueeze(1) * x_mean)
        detrended = series_2d - trend_line
        features["normalized_volatility"] = detrended.std(dim=1) / (series_2d.std(dim=1) + 1e-8)
        features["peak_location"] = series_2d.argmax(dim=1).float() / seq_len
        return features

    def calc_trend_reward(self, gen_features, target_meta):
        batch_size = len(target_meta)
        rewards = torch.zeros(batch_size, device=self.device)
        gen_slope = gen_features["normalized_slope"]
        hit = self.config["hit_score"]
        trend_soft_alpha = float(self.config.get("trend_soft_alpha", 0.3))
        trend_soft_alpha = min(max(trend_soft_alpha, 0.0), 1.0)

        for i in range(batch_size):
            target_label = target_meta[i]["trend"]
            slope = gen_slope[i].item()
            hard_score = 1.0 if classify_trend(slope) == target_label else 0.0
            soft_score = calc_trend_soft_score(slope, target_label)
            rewards[i] = hit * ((1.0 - trend_soft_alpha) * hard_score + trend_soft_alpha * soft_score)

        return rewards * self.config["weight_trend"]

    def calc_volatility_reward(self, gen_features, target_meta):
        batch_size = len(target_meta)
        rewards = torch.zeros(batch_size, device=self.device)
        gen_vol = gen_features["normalized_volatility"]
        hit = self.config["hit_score"]
        margin_ratio = max(float(self.config.get("volatility_margin_ratio", 0.15)), 1e-6)
        vol_bounds = {
            "low": (0.0, 0.5),
            "medium": (0.5, 0.8),
            "high": (0.8, 1.0),
        }
        for i in range(batch_size):
            lower, upper = vol_bounds.get(target_meta[i]["volatility"], (0.0, 1.0))
            width = upper - lower
            margin = width * margin_ratio
            center = 0.5 * (lower + upper)
            vol = gen_vol[i].item()
            if lower <= vol <= upper:
                if width <= 2 * margin:
                    reward = hit
                else:
                    distance_to_edge = min(vol - lower, upper - vol)
                    if distance_to_edge >= margin:
                        reward = hit
                    else:
                        reward = hit * (0.5 + 0.5 * distance_to_edge / margin)
            else:
                distance = lower - vol if vol < lower else vol - upper
                reward = hit * max(0.0, 0.5 * (1.0 - distance / max(width, 1e-6)))
            center_bonus = max(0.0, 1.0 - abs(vol - center) / max(width / 2.0, 1e-6))
            rewards[i] = min(hit, reward + 0.15 * hit * center_bonus)
        return rewards * self.config["weight_volatility"]

    def calc_peak_reward(self, gen_features, target_meta):
        batch_size = len(target_meta)
        rewards = torch.zeros(batch_size, device=self.device)
        gen_peak = gen_features["peak_location"]
        hit = self.config["hit_score"]
        peak_center = {"early": 1 / 6, "middle": 0.5, "late": 5 / 6}
        for i in range(batch_size):
            center = peak_center.get(target_meta[i]["peak_location"], 0.5)
            peak = gen_peak[i].item()
            rewards[i] = hit * max(0.0, 1.0 - abs(peak - center) / (1 / 3))
        return rewards * self.config["weight_peak"]

    def calc_mse_reward(self, generated, target):
        mse = F.mse_loss(generated, target, reduction="none").mean(dim=(1, 2))
        temperature = max(self.config["mse_temperature"], 1e-6)
        hit = self.config["hit_score"]
        mse_reward = hit * torch.exp(-mse / temperature)
        return mse_reward * self.config["weight_mse"]

    def smooth_compress(self, reward):
        if not self.config.get("smooth_clip_enabled", True):
            return reward
        scale = max(self.config.get("smooth_clip_scale", 15.0), 1e-6)
        clip_max = self.config.get("smooth_clip_max", 30.0)
        return torch.tanh(reward / scale) * clip_max

    def relative_normalize(self, reward):
        if reward.numel() <= 1 or not self.config.get("use_relative_reward", False):
            return reward
        centered = reward - reward.mean()
        normalized = centered / (reward.std(unbiased=False) + 1e-6)
        return normalized * self.config.get("relative_reward_scale", 5.0)

    def _compute_raw(self, series, target_meta, target_series):
        batch_size = series.shape[0]
        raw_total = torch.zeros(batch_size, device=self.device)
        info = {}
        gen_features = self.extract_features_from_series(series)

        if self.config["use_trend"] and target_meta is not None:
            r = self.calc_trend_reward(gen_features, target_meta)
            raw_total += r
            info["trend"] = r

        if self.config["use_volatility"] and target_meta is not None:
            r = self.calc_volatility_reward(gen_features, target_meta)
            raw_total += r
            info["volatility"] = r

        if self.config["use_peak"] and target_meta is not None:
            r = self.calc_peak_reward(gen_features, target_meta)
            raw_total += r
            info["peak"] = r

        if self.config["use_mse"] and target_series is not None:
            r = self.calc_mse_reward(series, target_series)
            raw_total += r
            info["mse"] = r

        return raw_total, info

    def get_raw_reward(self, series, target_meta=None, target_series=None):
        raw_total, info = self._compute_raw(series, target_meta, target_series)
        compressed = self.smooth_compress(raw_total)
        info["raw_total"] = raw_total
        info["compressed_total"] = compressed
        return compressed, info

    def get_reward(self, series, target_meta=None, target_series=None, return_details=False):
        raw_total, info = self._compute_raw(series, target_meta, target_series)
        compressed_total = self.smooth_compress(raw_total)
        final_total = self.relative_normalize(compressed_total)
        info["raw_total"] = raw_total
        info["compressed_total"] = compressed_total
        info["final_total"] = final_total
        if return_details:
            return final_total, info
        return final_total, info
