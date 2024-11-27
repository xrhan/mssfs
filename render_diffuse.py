import os
import cv2
import numpy as np
from numpy.linalg import norm
import math
import json


def load_normal_to_np(normal_file_path):
    """
    Load a normal map from file, resize and normalize it, and generate a mask.

    Args:
        normal_file_path (str): Path to the normal map file.

    Returns:
        tuple: Normal map (3D array) and mask (2D array).
    """  
    img = cv2.imread(normal_file_path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    resized_img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    normal_map = np.float32(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)) / 65535.0
    normal_map = 2 * normal_map - 1
    mask = np.abs(1 - np.sqrt(np.sum(normal_map**2, axis=2))) < 1.0e-3
    return normal_map, mask


def render_lambertian_shadowless(n, l, albedo, threshold = True):
    """
    Render a shadowless Lambertian surface given normals, light direction, and albedo.

    Args:
        n (np.ndarray): Normal map (3 x H x W).
        l (list or np.ndarray): Light direction (3D vector).
        albedo (float): Albedo coefficient.
        threshold (bool): Whether to threshold negative intensities.

    Returns:
        np.ndarray: Rendered intensity map.
    """
    mask = np.abs(1 - np.sqrt(np.sum(n**2, axis=0))) < 1.0e-3
    light_dir = l / norm(l)
    intensity_map = np.tensordot(n, light_dir, axes=1) * albedo
    if threshold:
        intensity_map[intensity_map < 0] = 0
    intensity_map[mask < 0.5] = 0
    return intensity_map


def random_on_sphere_constrained_fixed():
    """
    Sample a random vector on a sphere with constraints on the angle (cone around view direction).

    Returns:
        np.ndarray: A normalized 3D vector.
    """  
    threshold = math.cos(math.pi / 2.5)
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)

    while vec[2] < threshold:
        vec = np.random.randn(3)
        vec /= np.linalg.norm(vec)

    return vec


def normalize_normals(normals, mask):
    # Compute norms of the normals (H x W)
    norms = np.linalg.norm(normals, axis=0)
    # Avoid division by zero by masking invalid regions
    norms = np.where(mask, norms, 1)
    # Normalize normals only where the mask is valid
    unit_normals = normals / norms
    # Zero out invalid regions
    unit_normals *= mask
    return unit_normals


def render_from_file(filename):
    """
    Perform Lambertian rendering from a normal map file.

    Args:
        filename (str): Path to the normal map file.

    Returns:
        tuple: Normal map, mask, albedo, light direction, and rendered image.
    """
    normal_map, mask = load_normal_to_np(filename)
    normal_map = np.moveaxis(normal_map, -1, 0)  # Convert to 3 x H x W format
    albedo = np.random.uniform(0.5, 1.0)
    light_dir = random_on_sphere_constrained_fixed()
    rendered_img = render_lambertian_shadowless(normal_map, light_dir, albedo, True)
    return normal_map, mask, albedo, light_dir, rendered_img
