# superres.py
import cv2
import numpy as np
import torch
from PIL import Image

def apply_gfpgan(image, upscale_factor):
    """
    Enhance the image using GFPGAN.
    :param image: NumPy array in BGR format.
    :param upscale_factor: Upscale factor (float).
    :return: Enhanced image as a NumPy array in BGR format.
    """
    from gfpgan import GFPGANer
    model_path = "./checkpoints/GFPGANv1.3.pth"
    gfpganer = GFPGANer(model_path=model_path, upscale=upscale_factor, arch='clean', channel_multiplier=2)
    _, _, restored_img = gfpganer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
    return restored_img

def apply_codeformer(image, upscale_factor):
    """
    Enhance the image using CodeFormer.
    :param image: NumPy array in BGR format.
    :param upscale_factor: Upscale factor (float).
    :return: Enhanced image as a NumPy array in BGR format.
    """
    from basicsr.archs.codeformer_arch import CodeFormer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CodeFormer()
    ckpt = torch.load("./checkpoints/CodeFormer.pth", map_location=device)
    model.load_state_dict(ckpt["params_ema"])
    model.to(device)
    model.eval()

    # Convert image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor, w=0.7)[0]
    output = output.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    output = (output * 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output_bgr

def apply_superres(image, upscale_factor, method):
    """
    Dispatch to the appropriate superresolution method.
    :param image: NumPy array in BGR format.
    :param upscale_factor: Upscale factor (float).
    :param method: "GFPGAN", "CodeFormer", or "GFPGAN,CodeFormer"
    :return: Enhanced image as a NumPy array in BGR format.
    """
    method = method.lower()
    if method == "gfpgan":
        return apply_gfpgan(image, upscale_factor)
    elif method == "codeformer":
        return apply_codeformer(image, upscale_factor)
    elif method == "gfpgan,codeformer":
        intermediate = apply_gfpgan(image, upscale_factor)
        final = apply_codeformer(intermediate, upscale_factor)
        return final
    else:
        return image
