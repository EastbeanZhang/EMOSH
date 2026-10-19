from torch import nn


class Extra_Animate_Adapter(nn.Module):
    def __init__(self, dim=5120, in_dim=16, patch_size=(1, 2, 2)):
        super().__init__()
        self.rendering_patch_embedding = nn.Conv3d(
            in_dim,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def rendering_inject(self, x, rendering_latents, extra_refer_num=0):
        for features, latent in zip(x, rendering_latents):
            control = self.rendering_patch_embedding(latent.unsqueeze(0))
            features[:, :, 1 + extra_refer_num :] += control
        return x

    def init_weights(self, wan_blocks, patch_embedding):
        self.rendering_patch_embedding.load_state_dict(patch_embedding.state_dict())
