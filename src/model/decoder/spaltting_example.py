import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_knn._C import distCUDA2
from .utils import get_covariance

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from core.options import Options

import kiui

class GaussianRenderer:
    def __init__(self, opt: Options):

        self.opt = opt
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

        # intrinsics

        self.tan_half_fov = np.tan(0.5 * self.opt.FoVy)

    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None, scale_modifier=0.5):
        device = gaussians['position'].device
        B, V = cam_view.shape[:2]
        # loop of loop...
        images = []
        alphas = []

        position_batch = gaussians['position']
        opacity_batch = gaussians['opacity']
        scale_batch = gaussians['scale']
        cov3D_batch = gaussians['cov3d']
        rgb_batch = gaussians['rgb']

        for b in range(B):
            # pos, opacity, scale, rotation, shs
            means3D = position_batch[b].contiguous().float()
            opacity = opacity_batch[b].contiguous().float()
            scales = scale_batch[b].contiguous().float()
            cov3D = cov3D_batch[b].contiguous().float()
            rgbs = rgb_batch[b].contiguous().float() # [N, 3]

            dist2 = torch.clamp_min(distCUDA2(means3D), 0.0000001)
            scales_ = torch.sqrt(dist2)[...,None].repeat(1, 3).detach()
            scale = (scales+1)*scales_
            cov3D = get_covariance(scale, cov3D).reshape(-1, 6)

            for v in range(V):

                # render novel views
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()

                raster_settings = GaussianRasterizationSettings(
                    image_height=self.opt.output_size_h,
                    image_width=self.opt.output_size_w,
                    tanfovx=self.tan_half_fov,
                    tanfovy=self.tan_half_fov,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                with torch.cuda.amp.autocast(enabled=True): 
                    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                        means3D=means3D,
                        means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                        shs=None,
                        colors_precomp=rgbs,
                        opacities=opacity,
                        # scales=None,
                        # rotations=None,
                        cov3D_precomp=cov3D,
                    )
                rendered_image = rendered_image.clamp(0, 1)
                #torch.cuda.synchronize()
                # images.append(rendered_image.cpu())
                # alphas.append(rendered_alpha.cpu())
                images.append(rendered_image)
                alphas.append(rendered_alpha)

        images = torch.stack(images, dim=0).view(B, V, 3, self.opt.output_size_h, self.opt.output_size_w)
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, self.opt.output_size_h, self.opt.output_size_w)

        return {
            "image": images.to(device), # [B, V, 3, H, W]
            "alpha": alphas.to(device), # [B, V, 1, H, W]
        }


    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement

        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = kiui.op.inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)

    def load_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians

    def load_gaussians_from_ply(self, path):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis = 1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        positions =  torch.tensor(xyz, dtype = torch.float)
        colors =  torch.tensor(SH2RGB(features_dc)[:, [2, 1, 0]], dtype = torch.float).squeeze(-1)
        opacity =  torch.sigmoid(torch.tensor(opacities, dtype = torch.float))
        scales =  torch.exp(torch.tensor(scales, dtype = torch.float))
        rotations =  torch.nn.functional.normalize(torch.tensor(rots, dtype = torch.float))

        gaussians = torch.cat([positions, opacity, scales, rotations, colors], dim=1)

        return gaussians

C0 = 0.28209479177387814
def SH2RGB(sh):
    return sh * C0 + 0.5