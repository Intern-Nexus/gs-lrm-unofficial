from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple
from jaxtyping import Float
from torch import Tensor, nn

from .transformer_processor.processor import Processor

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ..types import Gaussians

from .encoder import Encoder


EPS = 1e-8

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions,
    eps = 1e-8,
):
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)

def build_covariance(
    scale,
    rotation_xyzw,
):
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )



@dataclass
class TransformerCfg:
    head_dim: int
    num_layers: int

@dataclass
class GaussianCfg:
    sh_degree: int
    scale_bias: float
    scale_max: float
    opacity_bias: float
    near_plane: float
    far_plane: float

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int

@dataclass
class EncoderLRMCfg:
    name: Literal["lrm"]
    patch_size: int
    attn_dim: int
    transformer: TransformerCfg
    gaussians_params: GaussianCfg
    apply_bounds_shim: bool
    near_disparity: float

class EncoderLRM(Encoder[EncoderLRMCfg]):
    tokenizer: nn.Sequential
    attn_processor: Processor
    token_decoder: nn.Sequential


    def __init__(self, cfg: EncoderLRMCfg) -> None:
        super().__init__(cfg)
        input_dim = 9 # RGB + plucker ray
        self.patch_size = cfg.patch_size
        self.attn_dim = cfg.attn_dim
        self.tokenizer = nn.Sequential(
            nn.Linear(input_dim * self.patch_size ** 2, self.attn_dim, bias=False),
        ) 
        self.tokenizer.apply(_init_weights)

        self.attn_processor = Processor(cfg)

        self.token_decoder = nn.Sequential(
            nn.LayerNorm(self.attn_dim, bias=False),
            nn.Linear(
                self.attn_dim, (1 + (cfg.gaussians_params.sh_degree + 1) ** 2 * 3 + 3 + 4 + 1) * self.patch_size ** 2,
                bias=False,
            )
        )
        self.token_decoder.apply(_init_weights)



    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))
    
    def get_camera_ray_dir(
        self,
        context: dict,
    ) -> Tuple[Tensor, Tensor]:

        dtype = context["image"].dtype
        device = context["image"].device
        B, V, _, H, W = context["image"].shape
        input_c2ws, input_intr_raw = context["extrinsics"], context["intrinsics"] #W2C

        # Reshape the intrinsics
        fx = input_intr_raw[..., 0, 0].unsqueeze(-1) * W # (B, V, 1)
        fy = input_intr_raw[..., 1, 1].unsqueeze(-1) * H # (B, V, 1)
        cx = input_intr_raw[..., 0, 2].unsqueeze(-1) * W # (B, V, 1)
        cy = input_intr_raw[..., 1, 2].unsqueeze(-1) * H # (B, V, 1)

        input_intr = torch.cat([fx, fy, cx, cy], dim=-1)

        # Embed camera info
        ray_o = input_c2ws[:, :, :3, 3].unsqueeze(2).expand(-1, -1, H * W, -1).float() # (B, V, H*W, 3) # camera origin
        x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        x = (x.to(dtype) + 0.5).view(1, 1, -1).expand(B, V, -1).to(device).contiguous()
        y = (y.to(dtype) + 0.5).view(1, 1, -1).expand(B, V, -1).to(device).contiguous()
        # unproject to camera space
        # breakpoint()
        x = (x - input_intr[:, :, 2:3]) / input_intr[:, :, 0:1] # 
        y = (y - input_intr[:, :, 3:4]) / input_intr[:, :, 1:2] # 
        ray_d = torch.stack([x, y, torch.ones_like(x)], dim=-1).float() # (B, V, H*W, 3)
        ray_d = F.normalize(ray_d, p=2, dim=-1)
        ray_d = ray_d @ input_c2ws[:, :, :3, :3].transpose(-1, -2).contiguous() # (B, V, H*W, 3)
        return ray_o, ray_d
    
    def feat2gaussian(
        self,
        gaussian_params: dict,
    ) -> Gaussians:
        means, scales, rotations, sh_feature, opacities = gaussian_params['xyz'], gaussian_params['scale'], gaussian_params['rotation'], gaussian_params['sh_feature'], gaussian_params['opacity']
        covariances = build_covariance(scales, rotations)

        # breakpoint()
        return Gaussians(
            means=means.float(),
            covariances=covariances.float(),
            harmonics=sh_feature.permute(0, 1, 3, 2).contiguous().float(),
            opacities=opacities.squeeze(-1).float(),
            # Note: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            # scales=scales,
            # rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )
    

    def forward(
        self,
        context: dict,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        ray_o, ray_d = self.get_camera_ray_dir(context)
        
        input_image_cam = torch.cat([context["image"].view(b, v, 3, -1).permute(0, 1, 3, 2).contiguous() * 2 - 1, 
                                     torch.cross(ray_o, ray_d, dim=-1),
                                     ray_d], dim=-1) # (B, V, H*W, 9)
        
        # Pachify
        patch_size = self.patch_size
        hh = h // patch_size
        ww = w // patch_size
        input_image_cam = rearrange(input_image_cam, 
                                    "b v (hh ph ww pw) d -> b (v hh ww) (ph pw d)", 
                                    hh=hh, ww=ww, ph=patch_size, pw=patch_size)

        # Tokenize the input images
        image_tokens = self.tokenizer(input_image_cam) # (B, V*hh*ww, D)
        with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"): 
            image_tokens = self.attn_processor(image_tokens, use_checkpoint=True)
        
        # Decode token to gaussians
        gaussians = self.token_decoder(image_tokens) 
        gaussians = rearrange(gaussians, "b (v hh ww) (ph pw d) -> b (v hh ph ww pw) d", v=v, hh=hh, ww=ww, ph=patch_size, pw=patch_size)

        dist, feature, scale, rotation, opacity = torch.split(gaussians, [1, (self.cfg.gaussians_params.sh_degree + 1) ** 2 * 3, 3, 4, 1], dim=-1)
        feature = feature.view(b, v*h*w, (self.cfg.gaussians_params.sh_degree + 1) ** 2, 3).contiguous()


        # Activate gaussian parameters
        w = rearrange(dist.sigmoid(), "b (v n) c -> b v n c", v=v)
        xyz = ray_o + ray_d * (context['near'].view(b, v, 1, 1) * (1 - w) + context['far'].view(b, v, 1, 1) * (w))

        
        scale = torch.exp(scale + self.cfg.gaussians_params.scale_bias).clamp(max=self.cfg.gaussians_params.scale_max)

        opacity = (opacity + self.cfg.gaussians_params.opacity_bias).sigmoid() #  

        rotation = rotation / (rotation.norm(dim=-1,keepdim=True) + EPS)

        gaussian_params = dict(xyz=xyz.flatten(1, 2), scale=scale, opacity=opacity, rotation=rotation, sh_feature=feature)
        
       
        gaussians = self.feat2gaussian(gaussian_params)
        gaussian_params.update({"gaussians": gaussians})

        return self.feat2gaussian(gaussian_params)
        

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=16, # Hard code patch size for now
            )

            if self.cfg.apply_bounds_shim:
                _, _, _, h, w = batch["context"]["image"].shape
                near_disparity = self.cfg.near_disparity * min(h, w)
                batch = apply_bounds_shim(batch, near_disparity, 0.5)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return self.epipolar_transformer.epipolar_sampler
