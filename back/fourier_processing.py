import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class FourierVideoProcessor:
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        freq_threshold_ratio: float = 0.1,
        use_spatial_freq: bool = False,
        use_temporal_freq: bool = True
    ):
        self.image_size = image_size
        self.freq_threshold_ratio = freq_threshold_ratio
        self.use_spatial_freq = use_spatial_freq
        self.use_temporal_freq = use_temporal_freq
    
    def temporal_fourier_transform(
        self, 
        video_tensor: torch.Tensor, 
        return_spectrum: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len, num_views, C, H, W = video_tensor.shape
        video_reshaped = video_tensor.permute(1, 2, 3, 4, 0)
        video_reshaped = video_reshaped.reshape(num_views * C * H * W, seq_len)
        
        freq_spectrum = torch.fft.fft(video_reshaped, dim=-1)
        freq_spectrum = torch.fft.fftshift(freq_spectrum, dim=-1)
        
        if return_spectrum:
            return freq_spectrum
        
        return freq_spectrum
    
    def separate_motion_static(
        self, 
        freq_spectrum: torch.Tensor, 
        seq_len: int,
        threshold_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if threshold_ratio is None:
            threshold_ratio = self.freq_threshold_ratio
        
        center_idx = seq_len // 2
        threshold = int(seq_len * threshold_ratio)
        
        low_freq_mask = torch.zeros_like(freq_spectrum, dtype=torch.bool)
        low_freq_mask[:, center_idx - threshold:center_idx + threshold + 1] = True
        
        high_freq_mask = ~low_freq_mask
        
        low_freq_spectrum = freq_spectrum * low_freq_mask
        high_freq_spectrum = freq_spectrum * high_freq_mask
        
        return low_freq_spectrum, high_freq_spectrum
    
    def inverse_temporal_fourier(
        self, 
        freq_spectrum: torch.Tensor, 
        seq_len: int
    ) -> torch.Tensor:
        freq_spectrum = torch.fft.ifftshift(freq_spectrum, dim=-1)
        spatial_data = torch.fft.ifft(freq_spectrum, n=seq_len, dim=-1)
        
        return spatial_data.real
    
    def reconstruct_from_freq(
        self, 
        freq_spectrum: torch.Tensor, 
        original_shape: Tuple[int, int, int, int, int]
    ) -> torch.Tensor:
        seq_len, num_views, C, H, W = original_shape
        data = self.inverse_temporal_fourier(freq_spectrum, seq_len)
        data = data.reshape(num_views, C, H, W, seq_len)
        data = data.permute(4, 0, 1, 2, 3)
        
        return data
    
    def process_video_temporal(
        self, 
        video_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = video_tensor.shape
        
        freq_spectrum = self.temporal_fourier_transform(video_tensor)
        low_freq_spectrum, high_freq_spectrum = self.separate_motion_static(
            freq_spectrum, original_shape[0]
        )
        
        static_component = self.reconstruct_from_freq(low_freq_spectrum, original_shape)
        motion_component = self.reconstruct_from_freq(high_freq_spectrum, original_shape)
        
        return static_component, motion_component
    
    def spatial_fourier_transform(
        self, 
        frame_tensor: torch.Tensor
    ) -> torch.Tensor:
        B, C, H, W = frame_tensor.shape
        freq_spectrum = torch.fft.fft2(frame_tensor, dim=(-2, -1))
        freq_spectrum = torch.fft.fftshift(freq_spectrum, dim=(-2, -1))
        
        return freq_spectrum
    
    def process_video_spatiotemporal(
        self, 
        video_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len, num_views, C, H, W = video_tensor.shape
        
        static_components = []
        motion_components = []
        
        for view_idx in range(num_views):
            view_video = video_tensor[:, view_idx]
            
            freq_spectrum = torch.fft.fftn(view_video, dim=(0, 1, 2))
            freq_spectrum = torch.fft.fftshift(freq_spectrum, dim=(0, 1, 2))
            
            freq_threshold = int(min(seq_len, H, W) * self.freq_threshold_ratio)
            
            center_t, center_h, center_w = seq_len // 2, H // 2, W // 2
            
            low_mask = torch.zeros_like(freq_spectrum, dtype=torch.bool)
            t_start = max(0, center_t - freq_threshold)
            t_end = min(seq_len, center_t + freq_threshold + 1)
            h_start = max(0, center_h - freq_threshold)
            h_end = min(H, center_h + freq_threshold + 1)
            w_start = max(0, center_w - freq_threshold)
            w_end = min(W, center_w + freq_threshold + 1)
            
            low_mask[t_start:t_end, h_start:h_end, w_start:w_end] = True
            high_mask = ~low_mask
            
            low_freq_spectrum = freq_spectrum * low_mask
            high_freq_spectrum = freq_spectrum * high_mask
            
            static_view = torch.fft.ifftn(torch.fft.ifftshift(low_freq_spectrum, dim=(0, 1, 2)), s=(seq_len, H, W)).real
            motion_view = torch.fft.ifftn(torch.fft.ifftshift(high_freq_spectrum, dim=(0, 1, 2)), s=(seq_len, H, W)).real
            
            static_components.append(static_view)
            motion_components.append(motion_view)
        
        static_component = torch.stack(static_components, dim=1)
        motion_component = torch.stack(motion_components, dim=1)
        
        return static_component, motion_component
    
    def __call__(
        self, 
        video_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_spatial_freq and self.use_temporal_freq:
            return self.process_video_spatiotemporal(video_tensor)
        else:
            return self.process_video_temporal(video_tensor)


class FourierFeatureExtractor:
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        num_freq_bands: int = 8,
        include_phase: bool = True
    ):
        self.image_size = image_size
        self.num_freq_bands = num_freq_bands
        self.include_phase = include_phase
    
    def extract_temporal_features(
        self, 
        video_tensor: torch.Tensor
    ) -> torch.Tensor:
        seq_len, num_views, C, H, W = video_tensor.shape
        
        freq_spectrum = torch.fft.fft(
            video_tensor.permute(1, 2, 3, 4, 0).reshape(num_views * C * H * W, seq_len),
            dim=-1
        )
        freq_spectrum = torch.fft.fftshift(freq_spectrum, dim=-1)
        
        band_size = seq_len // (2 * self.num_freq_bands)
        features = []
        
        for i in range(self.num_freq_bands):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size
            
            band_magnitude = torch.abs(freq_spectrum[:, start_idx:end_idx])
            band_mean = band_magnitude.mean(dim=-1)
            band_max = band_magnitude.max(dim=-1)[0]
            
            features.append(band_mean)
            features.append(band_max)
            
            if self.include_phase:
                band_phase = torch.angle(freq_spectrum[:, start_idx:end_idx])
                band_phase_mean = band_phase.mean(dim=-1)
                features.append(band_phase_mean)
        
        features = torch.cat(features, dim=-1)
        features = features.reshape(num_views, C, H, W, -1)
        features = features.permute(4, 0, 1, 2, 3)
        
        return features


def normalize_frequency_masking(
    video: torch.Tensor,
    threshold_ratio: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    processor = FourierVideoProcessor(
        freq_threshold_ratio=threshold_ratio,
        use_temporal_freq=True
    )
    static, motion = processor(video)
    return static, motion
