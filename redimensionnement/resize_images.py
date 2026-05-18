import argparse
import os

import cv2


def resize_images_in_folder(input_folder, output_folder, img_size=(240, 306)):
    """
    Resize images from input_folder and save them to output_folder.

    img_size is (height, width). OpenCV receives the reversed order.
    """
    os.makedirs(output_folder, exist_ok=True)

    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    print(f"Resizing images from '{input_folder}' to '{output_folder}'...")

    for filename in sorted(os.listdir(input_folder)):
        if not filename.lower().endswith(supported_formats):
            continue

        img_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not load image: {filename}")
                continue

            resized_img = cv2.resize(
                img,
                (img_size[1], img_size[0]),
                interpolation=cv2.INTER_AREA,
            )
            cv2.imwrite(output_path, resized_img)
        except Exception as exc:
            print(f"Error processing {filename}: {exc}")

    print(f"Finished resizing images in '{input_folder}'.")


def main():
    parser = argparse.ArgumentParser(description="Resize the local blood-smear image dataset.")
    parser.add_argument(
        "--input-root",
        default=os.path.join("redimensionnement", "Original_images"),
        help="Root folder containing Original_images_healthy and Original_images_anemic.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join("redimensionnement", "Original_images_240x306"),
        help="Root folder where resized healthy/anemic folders will be written.",
    )
    parser.add_argument("--height", type=int, default=240, help="Output image height.")
    parser.add_argument("--width", type=int, default=306, help="Output image width.")
    args = parser.parse_args()

    healthy_input_folder = os.path.join(args.input_root, "Original_images_healthy")
    anemic_input_folder = os.path.join(args.input_root, "Original_images_anemic")

    healthy_output_folder = os.path.join(args.output_root, "healthy")
    anemic_output_folder = os.path.join(args.output_root, "anemic")

    print(f"Starting image resizing process to {args.output_root}...")
    resize_images_in_folder(
        healthy_input_folder,
        healthy_output_folder,
        img_size=(args.height, args.width),
    )
    resize_images_in_folder(
        anemic_input_folder,
        anemic_output_folder,
        img_size=(args.height, args.width),
    )
    print("Image resizing process completed.")


if __name__ == "__main__":
    main()
