"""
Enhanced Feature Extraction with Semantic Segmentation
Addresses: Background contamination issue in color extraction
Author: Based on advisor feedback for fashion-volatility research
"""

import numpy as np
import cv2
import torch
from torchvision import models, transforms
from sklearn.cluster import KMeans
from PIL import Image
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")


class FashionFeatureExtractor:
    def __init__(self, use_segmentation=True):
        """
        Initialize feature extractor with optional semantic segmentation

        Args:
            use_segmentation: If True, isolate clothing before color extraction
        """
        self.use_segmentation = use_segmentation

        if use_segmentation:
            # Load pre-trained segmentation model
            self.seg_model = models.segmentation.deeplabv3_resnet50(pretrained=True)
            self.seg_model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_person_mask(self, img):
        """
        Extract person/clothing mask using semantic segmentation

        Args:
            img: numpy array (H, W, 3) in RGB

        Returns:
            Binary mask where 1 = person, 0 = background
        """
        # Prepare image
        input_tensor = self.transform(Image.fromarray(img)).unsqueeze(0)

        # Run segmentation
        with torch.no_grad():
            output = self.seg_model(input_tensor)["out"][0]

        # DeepLabV3 class 15 = person
        person_mask = output.argmax(0).numpy() == 15

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        person_mask = cv2.morphologyEx(
            person_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        )

        return person_mask.astype(bool)

    def extract_dominant_colors(self, pixels, n_colors=5):
        """
        Extract dominant colors using K-Means

        Args:
            pixels: Nx3 array of RGB values
            n_colors: Number of dominant colors to extract

        Returns:
            colors: n_colors x 3 array of RGB values
            proportions: Proportion of each color
        """
        if len(pixels) < n_colors:
            # Not enough pixels, pad with zeros
            colors = np.zeros((n_colors, 3))
            proportions = np.zeros(n_colors)
            return colors, proportions

        # Normalize to [0, 1]
        pixels_norm = pixels / 255.0

        # K-Means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels_norm)

        # Get colors and their proportions
        colors = kmeans.cluster_centers_ * 255
        labels = kmeans.labels_

        proportions = np.array(
            [np.sum(labels == i) / len(labels) for i in range(n_colors)]
        )

        # Sort by proportion (most dominant first)
        sort_idx = np.argsort(proportions)[::-1]
        colors = colors[sort_idx]
        proportions = proportions[sort_idx]

        return colors, proportions

    def rgb_to_hsv(self, rgb):
        """
        Convert RGB [0–255] to HSV (OpenCV scale)
        H: [0,179], S: [0,255], V: [0,255]
        """
        rgb_uint8 = np.uint8([[rgb]])
        h, s, v = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)[0, 0]
        return h, s, v

    def calculate_vibrancy(self, colors, proportions):
        """
        Calculate overall vibrancy score from dominant colors

        Vibrancy = weighted average of saturation * value (brightness)
        Higher vibrancy = more saturated, bright colors

        Args:
            colors: Nx3 array of RGB values
            proportions: N-length array of color proportions

        Returns:
            vibrancy_score: Float in [0, 1]
        """
        vibrancy_scores = []

        for color in colors:
            h, s, v = self.rgb_to_hsv(color)
            # Vibrancy = saturation * brightness
            # (normalized to [0, 1])
            vibrancy = (s / 255.0) * (v / 255.0)
            vibrancy_scores.append(vibrancy)

        # Weighted average by proportion
        vibrancy_score = np.sum(np.array(vibrancy_scores) * proportions)

        return vibrancy_score

    def extract_features(self, img_path, debug=False):
        """
        Main extraction pipeline

        Args:
            img_path: Path to runway image
            debug: If True, save intermediate visualizations

        Returns:
            features: Dictionary with all extracted features
        """

        # ---------- 1. Validate path ----------
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # ---------- 2. Load image safely ----------
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"OpenCV failed to load image: {img_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ---------- 3. Get clothing pixels ----------
        if self.use_segmentation:
            person_mask = self.get_person_mask(img_rgb)
            clothing_pixels = img_rgb[person_mask]

            # Fallback if segmentation fails (important for runway shots)
            if clothing_pixels.size == 0:
                clothing_pixels = img_rgb.reshape(-1, 3)
            else:
                clothing_pixels = clothing_pixels.reshape(-1, 3)

            if debug:
                mask_viz = np.zeros_like(img_rgb)
                mask_viz[person_mask] = img_rgb[person_mask]
                debug_path = img_path.replace(".jpg", "_mask.jpg")
                cv2.imwrite(debug_path, cv2.cvtColor(mask_viz, cv2.COLOR_RGB2BGR))
        else:
            clothing_pixels = img_rgb.reshape(-1, 3)

        # ---------- 4. Extract dominant colors ----------
        colors, proportions = self.extract_dominant_colors(clothing_pixels)

        # ---------- 5. Calculate vibrancy ----------
        vibrancy = self.calculate_vibrancy(colors, proportions)

        # ---------- 6. Package features ----------
        features = {
            "vibrancy": vibrancy,
            "dominant_colors": colors,
            "color_proportions": proportions,
            "n_pixels": len(clothing_pixels),
            "mean_saturation": np.mean([self.rgb_to_hsv(c)[1] for c in colors]),
            "mean_brightness": np.mean([self.rgb_to_hsv(c)[2] for c in colors]),
        }

        return features


# Example usage
if __name__ == "__main__":
    extractor = FashionFeatureExtractor(use_segmentation=True)

    # Test on single image
    features = extractor.extract_features(
        "../data/raw_images/paris_complete/alexander_mcqueen/fall_2016_ready_to_wear/alexander-mcqueen-fall-2016-ready-to-wear-0.png",
        debug=True,
    )
    image_dir = Path("../data/raw_images/paris_complete/chanel")

    print(f"Vibrancy Score: {features['vibrancy']:.3f}")
    print("Dominant Colors (RGB):")

    for i, (color, prop) in enumerate(
        zip(features["dominant_colors"], features["color_proportions"])
    ):
        rgb = tuple(int(c) for c in color)
        print(f"  Color {i + 1}: {rgb} ({prop * 100:.1f}%)")
