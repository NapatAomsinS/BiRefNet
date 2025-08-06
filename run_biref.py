import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
from image_proc import refine_foreground
from models.birefnet import BiRefNet
from PIL import Image
from torchvision import transforms


def main(section_num=2):
    parser = argparse.ArgumentParser(description="Background removal layer")
    parser.add_argument("--input_dir", required=True, help="Input directory path")
    parser.add_argument("--output_dir", required=True, help="Output directory path")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"\n------- Started Section: {section_num} --------")
    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print("-----------------------------------")

    # Loading Biref -> Probably should modify this to load from local disk later
    birefnet = BiRefNet.from_pretrained(
        [
            "zhengpeng7/BiRefNet",
            "zhengpeng7/BiRefNet-portrait",
            "zhengpeng7/BiRefNet-legacy",
            "zhengpeng7/BiRefNet-DIS5K-TR_TEs",
            "zhengpeng7/BiRefNet-DIS5K",
            "zhengpeng7/BiRefNet-HRSOD",
            "zhengpeng7/BiRefNet-COD",
            "zhengpeng7/BiRefNet_lite",  # Modify the `bb` in `config.py` to `swin_v1_tiny`.
        ][0]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision(["high", "highest"][0])

    birefnet.to(device)
    birefnet.eval()
    print("BiRefNet is loaded and ready to use.")

    # Input Data
    transform_image = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    autocast_ctx = torch.amp.autocast(
        device_type="cuda", dtype=[torch.float16, torch.bfloat16][0]
    )
    src_dir = input_dir
    image_paths = sorted(glob(os.path.join(src_dir, "*")))
    dst_dir = output_dir / "predictions"
    masked_dir = output_dir / "masked_image"
    result_dir = output_dir / "result_image"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(masked_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    for image_path in image_paths[:]:
        print("Processing {} ...".format(image_path))
        image = Image.open(image_path)
        image = image.convert("RGB") if image.mode != "RGB" else image
        input_images = transform_image(image).unsqueeze(0).to(device)

        # Prediction
        with autocast_ctx, torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().to(torch.float32).cpu()
        pred = preds[0].squeeze()

        # Show Results
        pred_pil = transforms.ToPILImage()(pred)
        pred_pil.resize(image.size).save(image_path.replace(str(src_dir), str(dst_dir)))

        image_masked = refine_foreground(image, pred_pil)
        image_masked.putalpha(pred_pil.resize(image.size))
        image_masked.resize(image.size).save(
            image_path.replace(str(src_dir), str(masked_dir)).replace(".jpg", ".png")
        )

        array_foreground = np.array(image_masked)[:, :, :3].astype(np.float32)
        array_mask = (np.array(image_masked)[:, :, 3:] / 255).astype(np.float32)
        array_background = np.zeros_like(array_foreground)
        array_background[:, :, :] = (0, 0, 0)
        array_foreground_background = (
            array_foreground * array_mask + array_background * (1 - array_mask)
        ).astype(np.uint8)
        Image.fromarray(array_foreground_background).save(
            image_path.replace(str(src_dir), str(result_dir))
        )

    print(f"------- Completed Section: {section_num} ------")


if __name__ == "__main__":
    main(section_num=2)
