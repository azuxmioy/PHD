"""SHAPify command line entry point."""

from .config import DEFAULT_FOCAL, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH, ShapeFitConfig
from .runner import fit_measured_betas
from .runner import main


def fit_betas(
    body_model,
    device,
    init_pose,
    init_betas,
    init_cam,
    openpose_joints,
    shoulder_width,
    target_height,
    target_mass,
    focal_length=None,
    image_width=None,
    image_height=None,
    mass_loss_weight=10.0,
    shoulder_loss_weight=1.0,
    height_loss_weight=100.0,
    beta_reg_weight=0.1,
    config=ShapeFitConfig(),
):
    """Compatibility wrapper for the measured SHAPify beta fitter."""

    camera = {
        "focal": DEFAULT_FOCAL if focal_length is None else focal_length,
        "width": DEFAULT_IMAGE_WIDTH if image_width is None else image_width,
        "height": DEFAULT_IMAGE_HEIGHT if image_height is None else image_height,
    }
    loss = {
        "mass_loss_weight": mass_loss_weight,
        "shoulder_loss_weight": shoulder_loss_weight,
        "height_loss_weight": height_loss_weight,
        "beta_reg_weight": beta_reg_weight,
    }
    return fit_measured_betas(
        body_model,
        device,
        init_pose,
        init_betas,
        init_cam,
        openpose_joints,
        shoulder_width,
        target_height,
        target_mass,
        camera,
        loss,
        config,
    )


if __name__ == "__main__":
    main()
