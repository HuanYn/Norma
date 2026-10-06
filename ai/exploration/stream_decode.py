"""Incremental causal Wan VAE decode, matching pinned upstream per-latent loop.

CC BY-NC-SA 4.0; same attribution and license as runtime.py. This retains
decoder feature caches, not independent decoding of isolated video chunks.
"""


class StreamingVAEDecoder:
    def __init__(self, vae):
        self.vae = vae
        vae.model.clear_cache()
        self.cache = [None] * vae.model._conv_num
        self.latent_count = 0

    def append(self, latents):
        import torch

        if latents.ndim != 4 or latents.shape[1] < 1:
            raise ValueError("Expected nonempty [C,T,H,W] latents")
        vae = self.vae
        with torch.inference_mode(), torch.autocast("cuda", dtype=vae.dtype):
            z = latents.unsqueeze(0)
            z = z / vae.scale[1].view(1, vae.model.z_dim, 1, 1, 1) + vae.scale[0].view(
                1, vae.model.z_dim, 1, 1, 1
            )
            x = vae.model.conv2(z)
            frames = [
                vae.model.decoder(
                    x[:, :, i : i + 1], feat_cache=self.cache, feat_idx=[0]
                )
                for i in range(x.shape[2])
            ]
            result = torch.cat(frames, dim=2).float().clamp_(-1, 1).squeeze(0)
        expected = 4 * latents.shape[1] - (3 if self.latent_count == 0 else 0)
        if result.shape[1] != expected or not bool(torch.isfinite(result).all()):
            raise RuntimeError("Invalid streaming VAE frame count or pixels")
        self.latent_count += latents.shape[1]
        return result

    def close(self):
        self.cache.clear()
        self.latent_count = 0
