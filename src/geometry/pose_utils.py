import torch
from typing import Union, Tuple

def normalize(x):
    return x / torch.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(torch.cross(vec1_avg, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    
    m = torch.stack([vec0, vec1, vec2, pos], dim=1)
    
    m_4x4 = torch.cat([m, torch.tensor([[0, 0, 0, 1]], dtype=pos.dtype)], dim=0)
    return m_4x4

def poses_avg(poses):

    poses_3x4 = poses[:, :3, :4]
    center = poses_3x4[:, :, -1].mean(dim=0)
    
    vec2 = normalize(poses_3x4[:, :, 2].sum(dim=0))
    # breakpoint()
    up = poses_3x4[:, :, 1].sum(dim=0)
    c2w = viewmatrix(vec2, up, center)
    
    return c2w

def centerize_scale_poses(
    in_c2ws: torch.Tensor,
    frame_method: str = 'mean_cam',
    scale_range: Union[Tuple[float, float], None] = None,
    scene_scale_method: str = 'two_cam'
) -> Tuple[torch.Tensor, torch.Tensor]:

    in_c2ws = in_c2ws.clone()
    N, _, _ = in_c2ws.shape

    if frame_method == 'mean_cam':
        # bottom = torch.tensor([0, 0, 0, 1.0], device=in_c2ws.device).view(1, 4)
        apos = poses_avg(in_c2ws)  
        # apos = torch.cat([apos, bottom], dim=0).unsqueeze(0)
        # breakpoint()
        in_c2ws = torch.matmul(torch.inverse(apos), in_c2ws)
    # TODO: the following two method dosen't support yet!
    elif frame_method == 'first_cam':
        first_c2w = in_c2ws[0]
        in_c2ws = torch.matmul(torch.inverse(first_c2w), in_c2ws)
    elif frame_method == 'center':
        scene_center = (torch.max(in_c2ws[:, :3, 3], dim=0).values +
                        torch.min(in_c2ws[:, :3, 3], dim=0).values) / 2
        in_c2ws[:, :3, 3] = in_c2ws[:, :3, 3] - scene_center
    else:
        raise NotImplementedError(f"Unknown frame_method: {frame_method}")


    scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))

    if scene_scale_method == "two_cam":
        two_cam_dist = torch.linalg.norm(in_c2ws[0, :3, 3] - in_c2ws[1, :3, 3])
        scene_scale = 1.0 / (two_cam_dist + 0.01)
    elif scene_scale_method == "fix_range":
        if scale_range is None:
            raise ValueError("scale_range must be provided when scene_scale_method is 'fix_range'")
        min_scale, max_scale = scale_range
        random_scale = torch.rand(1, device=in_c2ws.device)[0] * (max_scale - min_scale) + min_scale
        scene_scale *= random_scale
    else:
        raise NotImplementedError(f"Unknown scene_scale_method: {scene_scale_method}")

    in_c2ws[:, :3, 3] = in_c2ws[:, :3, 3] / scene_scale

    return in_c2ws, scene_scale