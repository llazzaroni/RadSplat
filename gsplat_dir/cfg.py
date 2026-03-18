from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from typing_extensions import Literal, assert_never
from submodules.gsplat.gsplat.strategy import DefaultStrategy, MCMCStrategy

@dataclass
class Config:
    pt_path: str = ""
    # Base random seed. Effective seed is seed + local_rank.
    seed: int = 42
    # If True, enable deterministic torch/cudnn behavior (slower, but more reproducible).
    deterministic: bool = False
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None

    save_first_ckp: bool = False
    nerf_init: bool = True
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Optional downsample factor for nerf_sample_* images. If None, uses data_factor.
    nerf_samples_data_factor: Optional[int] = None
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # In dual-loader training, nerf batch size = batch_size * nerf_batch_factor.
    nerf_batch_factor: int = 2
    # In dual-loader training, constant weight for real-image loss.
    dual_real_loss_weight: float = 1.0
    # In dual-loader training, initial weight for nerf-sample loss (step 0).
    dual_nerf_loss_weight: float = 2.0
    # Exponential decay parameterization: after this many steps, nerf weight is quartered
    # (e.g. 2.0 -> 0.5).
    dual_nerf_decay_steps_to_quarter: int = 1000
    # Disable the nerf branch when the scheduled nerf weight falls below this threshold.
    dual_nerf_disable_threshold: float = 0.1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 5_000
    # Staged training: nerf-only phase steps (used by run_gsplat_staged.py).
    staged_nerf_phase_steps: int = 0
    # If True, staged pretraining phase uses both real + nerf samples
    # (with real images loaded at nerf_samples_data_factor resolution).
    staged_include_real_in_nerf_phase: bool = False
    # Staged training: real-only phase steps (used by run_gsplat_staged.py).
    staged_real_phase_steps: int = 5_000
    # If True, reset opacities to staged_reset_opacity_value at phase switch.
    staged_reset_opacity_after_nerf: bool = True
    # Opacity value to reset to at phase switch (sigmoid-space).
    staged_reset_opacity_value: float = 0.1
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # If True, real-image reconstruction term is L1; otherwise L2 (MSE).
    # NeRF-sample supervision remains weighted L2 in all cases.
    use_l1_for_real_samples: bool = False
    # If True, NeRF-sample supervision in dual runner uses weighted L1
    # instead of weighted L2.
    use_l1_for_nerf_samples: bool = False
    # Enable additional depth supervision from NeRF-generated depth maps.
    use_nerf_depth_supervision: bool = False
    # Root folder prefix under dataset for NeRF depth maps:
    # depths_nerf[, depths_nerf_2, depths_nerf_4, depths_nerf_8].
    nerf_depth_prefix: str = "depths_nerf"
    # Optional downsample factor for NeRF depth maps. If None, follows sample image factor.
    nerf_depth_data_factor: Optional[int] = None
    # Apply NeRF depth supervision on real-image batches.
    nerf_depth_include_real: bool = True
    # Apply NeRF depth supervision on nerf_sample batches.
    nerf_depth_include_nerf_samples: bool = True
    # Stop applying NeRF depth supervision after this global step.
    # Set < 0 to keep it active for all training.
    nerf_depth_max_steps: int = 1000
    # Weight of NeRF depth supervision term.
    nerf_depth_lambda: float = 1e-2
    # If True, compute NeRF depth loss in log-depth space.
    nerf_depth_log_space: bool = True
    # Optional initial warmup phase (in steps) for depth-guided optimization before
    # switching to standard dual training.
    depth_warmup_steps: int = 0
    # During depth warmup, disable the nerf-sample RGB branch and optimize only with
    # real-image branch (+ optional depth supervision).
    depth_warmup_disable_nerf_branch: bool = True
    # During depth warmup, disable real-image RGB branch and optimize only with
    # nerf-sample branch (+ optional depth supervision).
    depth_warmup_only_nerf_loss: bool = False
    # During depth warmup, optimize using only NeRF depth supervision loss
    # (no RGB real/nerf reconstruction term). Depth supervision is evaluated on
    # both real and nerf-sample batches when available.
    depth_warmup_only_depth_loss: bool = False
    # During depth warmup, force depth supervision active even if nerf_depth_max_steps
    # would otherwise disable it.
    depth_warmup_force_depth_supervision: bool = True

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = True
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)
