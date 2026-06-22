import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from skimage.transform import resize
from skimage.segmentation import flood_fill
import tifffile as tiff
import cv2
from typing import Union
import xenquaco.ilastik_workflow as ilastik_workflow
import xenquaco.data_processing as data_processing


def generate_mask(ilastik_program_path: Union[str, Path],
                  input_image_path: Union[str, Path],
                  pixel_classification_model_path: Union[str, Path],
                  object_classification_model_path: Union[str, Path],
                  pixel_classification_export_type: str = 'Probabilities',
                  mask_path: Union[str, Path] = None) -> None:
    """
    Runs pixel and object classification workflows to create and save binary mask

    Parameters
    ----------
    ilastik_program_path : str or Path
        Path to ilastik program
    input_image_path : str or Path
        Path to input image
    pixel_classification_model_path : str or Path
        Path to pixel classification model
    object_classification_model_path : str or Path
        Path to object classification model
    pixel_classification_export_type : str, optional
        Export type for pixel classification. Default is 'Probabilities'.
    mask_path : str or Path, optional
        Path at which to save mask. Default is None.
    """
    probability_map_path = Path(input_image_path).with_name(
        Path(input_image_path).name.replace('.tiff', '_probability_map.tiff'))
    pixel_workflow_args = ilastik_workflow.get_pixel_workflow_args(ilastik_program_path,
                                                                   pixel_classification_model_path,
                                                                   input_image_path, probability_map_path,
                                                                   pixel_classification_export_type)
    ilastik_workflow.run_ilastik_workflow(pixel_workflow_args)

    if mask_path is None:
        mask_path = Path(input_image_path).with_name(
            Path(input_image_path).name.replace('.tiff', '_mask.tiff'))
    object_workflow_args = ilastik_workflow.get_object_workflow_args(ilastik_program_path,
                                                                     object_classification_model_path,
                                                                     input_image_path, probability_map_path,
                                                                     mask_path)
    ilastik_workflow.run_ilastik_workflow(object_workflow_args)

    os.remove(probability_map_path)


def rounddown_10(x: Union[int, float]) -> int:
    """
    Helper function rounds down given number to nearest multiple of 10
    """
    return int(math.floor(x / 10.0)) * 10


def get_hist2d(transcripts_plot: pd.DataFrame, transcripts_bins: pd.DataFrame = None) -> tuple:
    """
    Generate a 2D histogram from a transcripts table using Xenium x_location/y_location columns

    Parameters
    ----------
    transcripts_plot : pd.DataFrame
        Transcripts table to plot
    transcripts_bins : pd.DataFrame, optional
        Transcripts table for histogram bins. Default is None.

    Returns
    -------
    img : np.ndarray
        Histogram of samples in x and y
    mask_x_bins : np.ndarray
        Array of bin edges along x axis
    mask_y_bins : np.ndarray
        Array of bin edges along y axis
    """
    if transcripts_bins is None:
        transcripts_bins = transcripts_plot

    transcripts_xy_plot = np.asarray(transcripts_plot[['x_location', 'y_location']])
    transcripts_xy_bins = np.asarray(transcripts_bins[['x_location', 'y_location']])

    img, mask_x_bins, mask_y_bins, _ = plt.hist2d(
        transcripts_xy_plot[:, 0],
        transcripts_xy_plot[:, 1],
        bins=[np.arange(min(rounddown_10(np.min(transcripts_xy_bins[:, 0])), 0),
                        int(np.ceil(np.max(transcripts_xy_bins[:, 0]))) + 10, 10),
              np.arange(min(rounddown_10(np.min(transcripts_xy_bins[:, 1])), 0),
                        int(np.ceil(np.max(transcripts_xy_bins[:, 1]))) + 10, 10)])
    plt.close()
    return img, mask_x_bins, mask_y_bins


def resize_mask(mask: np.ndarray, by: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """
    Resizes mask to share same dimensions as `by`

    Parameters
    ----------
    mask : np.ndarray
        Array to resize
    by : np.ndarray
        Array to resize `mask` to
    thresh : float, optional
        Threshold to assign 1 or 0. Default is 0.5.

    Returns
    -------
    resized_mask : np.ndarray
        Resized mask
    """
    height = by.shape[0]
    width = by.shape[1]
    resized_mask = resize(mask, (height, width), preserve_range=True)
    resized_mask = np.where(resized_mask >= thresh, 1, 0)
    return resized_mask


def dilate_array(image_array: np.ndarray, kernel_size: int, num_iterations: int = 1) -> np.ndarray:
    """Dilate input array by given kernel size"""
    matrix = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(image_array, matrix, iterations=num_iterations)


def erode_array(image_array: np.ndarray, kernel_size: int, num_iterations: int = 1) -> np.ndarray:
    """Erode input image array by kernel size"""
    matrix = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(image_array, matrix, iterations=num_iterations)


def compress_image(image: np.ndarray, bin_size: int) -> np.ndarray:
    """
    Compress image array by summing pixel intensity values within square bins

    Parameters
    ----------
    image : np.ndarray
        Input image array (must be 2D)
    bin_size : int
        Size of bins

    Returns
    -------
    binned_image : np.ndarray
        Binned image array
    """
    num_rows = image.shape[0] // bin_size
    num_cols = image.shape[1] // bin_size
    binned_image = image[:num_rows * bin_size, :num_cols * bin_size].reshape(
        num_rows, bin_size, num_cols, bin_size)
    return np.sum(binned_image, axis=(1, 3))


def normalize_intensities(image: np.ndarray) -> np.ndarray:
    """Normalize pixel intensity values to range (0, 255)"""
    min_value = np.min(image)
    max_value = np.max(image)
    normalized_image = (image - min_value) / (max_value - min_value) * 255
    return normalized_image.astype(np.uint8)


def on_tissue_threshold(dapi_image: np.ndarray, bin_count: int) -> int:
    """
    Approximates on-tissue intensity threshold from histogram of pixel intensity values
    """
    hist, bins = np.histogram(dapi_image, bins=np.linspace(0, 255, bin_count), density=False)
    max_bin_index = np.argmax(hist)
    upper_boundary_max_bin = bins[max_bin_index + 1]
    return np.ceil(upper_boundary_max_bin)


def create_transcripts_image(transcripts: pd.DataFrame,
                              transcripts_image_path: Union[str, Path] = '') -> tuple:
    """
    Creates transcripts image from transcripts via 2D histogram

    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcripts table (must include x_location, y_location columns)
    transcripts_image_path : str or Path, optional
        Path at which to save transcripts image. Default is ''.

    Returns
    -------
    img : np.ndarray
        Histogram of samples in x and y
    mask_x_bins : np.ndarray
        Array of bin edges in x
    mask_y_bins : np.ndarray
        Array of bin edges in y
    """
    img, mask_x_bins, mask_y_bins = get_hist2d(transcripts)

    if transcripts_image_path != '':
        tiff.imwrite(transcripts_image_path, img.T)

    return img, mask_x_bins, mask_y_bins


def generate_transcripts_mask(transcripts_image_path: Union[str, Path],
                               ilastik_program_path: Union[str, Path],
                               pixel_classification_model_path: Union[str, Path],
                               object_classification_model_path: Union[str, Path],
                               transcripts: pd.DataFrame = None) -> np.ndarray:
    """
    Return transcripts mask, generating from transcripts table if image does not exist

    Parameters
    ----------
    transcripts_image_path : str or Path
        Path to transcripts image
    ilastik_program_path : str or Path
        Path to ilastik program
    pixel_classification_model_path : str or Path
        Path to pixel classification model
    object_classification_model_path : str or Path
        Path to object classification model
    transcripts : pd.DataFrame, optional
        Transcripts table. Required if transcripts_image_path does not exist.

    Returns
    -------
    transcripts_mask : np.ndarray
        Transcripts mask array
    """
    if not os.path.exists(transcripts_image_path):
        if transcripts is not None:
            _ = create_transcripts_image(transcripts, transcripts_image_path)
        else:
            raise ValueError("transcripts_image_path does not exist; transcripts must be provided")

    generate_mask(ilastik_program_path, transcripts_image_path,
                  pixel_classification_model_path, object_classification_model_path)

    transcripts_mask_path = Path(transcripts_image_path).with_name(
        Path(transcripts_image_path).name.replace('.tiff', '_mask.tiff'))
    return tiff.imread(transcripts_mask_path)


def create_dapi_image(high_res_dapi_image_path: Union[str, Path],
                      low_res_dapi_image_path: Union[str, Path]) -> np.ndarray:
    """
    Creates a lower-resolution DAPI image by binning, normalizing, and removing off-tissue pixels.

    Handles multi-channel OME-TIFF images (e.g. Xenium morphology_focus.ome.tif) by extracting
    channel 0 (DAPI) when the image has more than 2 dimensions.

    Parameters
    ----------
    high_res_dapi_image_path : str or Path
        Path to high-resolution DAPI image
    low_res_dapi_image_path : str or Path
        Path at which to save low-resolution DAPI image

    Returns
    -------
    dapi_image : np.ndarray
        Array of low-resolution DAPI image
    """
    try:
        high_res_dapi_image = tiff.imread(high_res_dapi_image_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f'DAPI image not found at {high_res_dapi_image_path}: {e}')

    # Extract DAPI channel (channel 0) from multi-channel Xenium OME-TIFF
    if high_res_dapi_image.ndim > 2:
        high_res_dapi_image = high_res_dapi_image[0]

    dapi_image = compress_image(high_res_dapi_image, bin_size=100)
    dapi_image = normalize_intensities(dapi_image)
    dapi_image = np.clip(dapi_image * 4, 0, 255).astype(np.uint8)
    threshold = on_tissue_threshold(dapi_image, 20)
    dapi_image = np.where(dapi_image < threshold, 0, dapi_image)
    tiff.imwrite(low_res_dapi_image_path, dapi_image)

    return dapi_image


def generate_dapi_mask(dapi_image_path: Union[str, Path],
                       ilastik_program_path: Union[str, Path],
                       pixel_classification_model_path: Union[str, Path],
                       object_classification_model_path: Union[str, Path],
                       high_res_dapi_image_path: Union[str, Path] = '') -> np.ndarray:
    """
    Generate binary DAPI mask from compressed DAPI image or high-res DAPI image

    Parameters
    ----------
    dapi_image_path : str or Path
        Path to compressed DAPI image
    ilastik_program_path : str or Path
        Path to ilastik program
    pixel_classification_model_path : str or Path
        Path to pixel classification model
    object_classification_model_path : str or Path
        Path to object classification model
    high_res_dapi_image_path : str or Path, optional
        Path to high-resolution DAPI image. Default is ''.

    Returns
    -------
    dapi_mask : np.ndarray
        DAPI mask array
    """
    if not os.path.exists(dapi_image_path):
        if high_res_dapi_image_path == '':
            raise ValueError("Compressed DAPI image does not exist. high_res_dapi_image_path must be provided")
        create_dapi_image(high_res_dapi_image_path, dapi_image_path)

    generate_mask(ilastik_program_path, dapi_image_path,
                  pixel_classification_model_path, object_classification_model_path)

    dapi_mask_path = Path(dapi_image_path).with_name(
        Path(dapi_image_path).name.replace('.tiff', '_mask.tiff'))
    return tiff.imread(dapi_mask_path)


def generate_detachment_mask(transcripts_mask_path: Union[str, Path],
                              dapi_mask_path: Union[str, Path],
                              detachment_mask_path: Union[str, Path]) -> np.ndarray:
    """
    Generate and save tissue detachment mask by subtracting transcript mask from DAPI mask.

    In Xenium experiments, this captures regions where tissue has detached from the slide
    during imaging (DAPI signal present but no transcripts detected).

    Parameters
    ----------
    transcripts_mask_path : str or Path
        Path to binary transcript mask
    dapi_mask_path : str or Path
        Path to binary DAPI mask
    detachment_mask_path : str or Path
        Path at which to save detachment mask

    Returns
    -------
    detachment_mask : np.ndarray
        Detachment mask binary array
    """
    transcripts_mask = tiff.imread(transcripts_mask_path)
    dapi_mask = tiff.imread(dapi_mask_path)
    dapi_mask = resize_mask(dapi_mask, by=transcripts_mask)
    detachment_mask = dapi_mask - transcripts_mask
    detachment_mask[detachment_mask == -1] = 0
    tiff.imwrite(detachment_mask_path, detachment_mask)
    return detachment_mask


def create_ventricle_genes_image(ventricle_genes_image_path: Union[str, Path],
                                  dapi_mask_path: Union[str, Path],
                                  transcripts_mask_path: Union[str, Path],
                                  transcripts: pd.DataFrame,
                                  ventricle_genes: list = ["Crb2", "Glis3", "Inhbb",
                                                            "Naaa", "Cd24a", "Dsg2", "Hdc",
                                                            "Shroom3", "Vit", "Rgs12", "Trp73"],
                                  threshold: int = 2) -> np.ndarray:
    """
    Creates and saves image of ventricle genes superimposed on DAPI image

    Parameters
    ----------
    ventricle_genes_image_path : str or Path
        Path at which to save ventricle genes image
    dapi_mask_path : str or Path
        Path to DAPI mask
    transcripts_mask_path : str or Path
        Path to transcripts mask
    transcripts : pd.DataFrame
        Transcripts table (must include feature_name, x_location, y_location columns)
    ventricle_genes : list, optional
        List of ventricle marker genes
    threshold : int, optional
        Threshold to binarize ventricle gene density image. Default is 2.

    Returns
    -------
    dapi_ventricles : np.ndarray
        Ventricle genes image overlaid on DAPI mask
    """
    dapi_mask = tiff.imread(dapi_mask_path)
    transcripts_mask = tiff.imread(transcripts_mask_path)
    dapi_mask = resize_mask(dapi_mask, by=transcripts_mask)

    gene_density_maps = []
    ventricle_gene_ts = transcripts[transcripts['feature_name'].isin(ventricle_genes)]
    genes_in_panel = np.unique(ventricle_gene_ts['feature_name'])

    for gene in genes_in_panel:
        gene_ts = ventricle_gene_ts[ventricle_gene_ts['feature_name'] == gene]
        gene_2dhist, _, _ = get_hist2d(gene_ts, transcripts)
        gene_2dhist[gene_2dhist < threshold] = 0
        gene_2dhist[gene_2dhist >= threshold] = 1
        gene_density_maps.append(gene_2dhist)

    ventricles = np.logical_and.reduce(gene_density_maps).astype(np.float64)
    dapi_ventricles = ventricles.T + dapi_mask
    tiff.imwrite(ventricle_genes_image_path, dapi_ventricles)
    return dapi_ventricles


def generate_ventricle_mask(ventricle_genes_image_path: Union[str, Path],
                             dapi_mask_path: Union[str, Path],
                             transcripts_mask_path: Union[str, Path],
                             ilastik_program_path: Union[str, Path],
                             pixel_classification_model_path: Union[str, Path],
                             object_classification_model_path: Union[str, Path],
                             transcripts: pd.DataFrame,
                             ventricle_genes: list = ["Crb2", "Glis3", "Inhbb",
                                                      "Naaa", "Cd24a", "Dsg2", "Hdc",
                                                      "Shroom3", "Vit", "Rgs12", "Trp73"]) -> np.ndarray:
    """
    Creates and saves binary ventricle mask from transcripts and ventricle gene list

    Parameters
    ----------
    ventricle_genes_image_path : str or Path
        Path to ventricle genes image
    dapi_mask_path : str or Path
        Path to binary DAPI mask
    transcripts_mask_path : str or Path
        Path to binary transcripts mask
    ilastik_program_path : str or Path
        Path to ilastik program
    pixel_classification_model_path : str or Path
        Path to pixel classification model
    object_classification_model_path : str or Path
        Path to object classification model
    transcripts : pd.DataFrame
        Transcripts table
    ventricle_genes : list, optional
        List of ventricle marker genes

    Returns
    -------
    ventricle_mask : np.ndarray
        Ventricle mask
    """
    if not os.path.exists(ventricle_genes_image_path):
        create_ventricle_genes_image(ventricle_genes_image_path, dapi_mask_path,
                                     transcripts_mask_path, transcripts, ventricle_genes)

    generate_mask(ilastik_program_path, ventricle_genes_image_path,
                  pixel_classification_model_path, object_classification_model_path,
                  pixel_classification_export_type='probabilities stage 2')

    ventricle_mask_path = Path(ventricle_genes_image_path).with_name(
        Path(ventricle_genes_image_path).name.replace('.tiff', '_mask.tiff'))
    return tiff.imread(ventricle_mask_path)


def generate_damage_mask(damage_mask_path: Union[str, Path],
                          dapi_image_path: Union[str, Path],
                          dapi_mask_path: Union[str, Path],
                          transcripts_mask_path: Union[str, Path],
                          ventricle_mask_path: Union[str, Path]) -> np.ndarray:
    """
    Generate and save binary tissue damage mask

    Parameters
    ----------
    damage_mask_path : str or Path
        Path at which to save damage mask
    dapi_image_path : str or Path
        Path to DAPI image
    dapi_mask_path : str or Path
        Path to binary DAPI mask
    transcripts_mask_path : str or Path
        Path to binary transcript mask
    ventricle_mask_path : str or Path
        Path to binary ventricles mask

    Returns
    -------
    damage : np.ndarray
        Damage mask array
    """
    dapi_image = tiff.imread(dapi_image_path)
    dapi_mask = tiff.imread(dapi_mask_path)
    transcripts_mask = tiff.imread(transcripts_mask_path)
    ventricle_mask = tiff.imread(ventricle_mask_path)

    threshold = on_tissue_threshold(dapi_image, 4)
    dapi_image_binary = np.where(dapi_image < threshold, 0, dapi_image)
    dapi_image_binary = np.where(dapi_image_binary > threshold - 1, 1, dapi_image_binary)
    dapi_dilated = dilate_array(dapi_image_binary, 40)
    dapi_dilate_eroded = erode_array(dapi_dilated, 35)
    damage_ventricles1 = dapi_dilate_eroded - dapi_mask

    dapi_mask = np.pad(dapi_mask, 1, mode='constant', constant_values=0)
    inverted_dapi_mask = np.array([[1 if bit == 0 else 0 for bit in row] for row in dapi_mask])
    damage_ventricles2 = flood_fill(inverted_dapi_mask, (0, 0), 0)

    damage_ventricles1 = resize_mask(damage_ventricles1, by=transcripts_mask)
    damage_ventricles2 = resize_mask(damage_ventricles2, by=transcripts_mask)
    damage_ventricles = damage_ventricles1 + damage_ventricles2
    damage_ventricles[damage_ventricles > 1] = 1

    ventricle_mask = resize_mask(ventricle_mask, by=transcripts_mask)
    damage = damage_ventricles - ventricle_mask
    damage[damage == -1] = 0

    tiff.imwrite(damage_mask_path, damage)
    return damage


def resize_all_masks(transcripts_mask_input: Union[np.ndarray, str, Path],
                     dapi_mask_input: Union[np.ndarray, str, Path],
                     detachment_mask_input: Union[np.ndarray, str, Path],
                     ventricle_mask_input: Union[np.ndarray, str, Path],
                     damage_mask_input: Union[np.ndarray, str, Path],
                     save: bool = False) -> tuple:
    """
    Resize all masks to share the same dimensions as the transcripts mask

    Parameters
    ----------
    transcripts_mask_input : np.ndarray, str, or Path
    dapi_mask_input : np.ndarray, str, or Path
    detachment_mask_input : np.ndarray, str, or Path
    ventricle_mask_input : np.ndarray, str, or Path
    damage_mask_input : np.ndarray, str, or Path
    save : bool, optional
        Whether to save resized masks. Default is False.

    Returns
    -------
    tuple of np.ndarray
        transcripts_mask, dapi_mask, detachment_mask, ventricle_mask, damage_mask
    """
    transcripts_mask = data_processing.process_input(transcripts_mask_input)
    dapi_mask = data_processing.process_input(dapi_mask_input)
    detachment_mask = data_processing.process_input(detachment_mask_input)
    ventricle_mask = data_processing.process_input(ventricle_mask_input)
    damage_mask = data_processing.process_input(damage_mask_input)

    dapi_mask = resize_mask(dapi_mask, by=transcripts_mask)
    detachment_mask = resize_mask(detachment_mask, by=transcripts_mask)
    ventricle_mask = resize_mask(ventricle_mask, by=transcripts_mask)
    damage_mask = resize_mask(damage_mask, by=transcripts_mask)

    if save:
        try:
            tiff.imwrite(transcripts_mask_input, transcripts_mask)
            tiff.imwrite(dapi_mask_input, dapi_mask)
            tiff.imwrite(detachment_mask_input, detachment_mask)
            tiff.imwrite(ventricle_mask_input, ventricle_mask)
            tiff.imwrite(damage_mask_input, damage_mask)
        except Exception as e:
            raise Exception(f"Ensure paths to all masks are provided: {e}")

    return transcripts_mask, dapi_mask, detachment_mask, ventricle_mask, damage_mask


def classify_pixels(transcripts_mask_input: Union[np.ndarray, str, Path],
                    detachment_mask_input: Union[np.ndarray, str, Path],
                    ventricle_mask_input: Union[np.ndarray, str, Path] = None,
                    damage_mask_input: Union[np.ndarray, str, Path] = None,
                    full_classification_out_file: Union[str, Path] = None) -> np.ndarray:
    """
    Combine all masks and classify each pixel as damage, tissue, detachment, ventricle, or off-tissue.

    Pixel classes: 0=off-tissue, 1=damage, 2=tissue, 3=detachment, 4=ventricles

    Parameters
    ----------
    transcripts_mask_input : np.ndarray, str, or Path
    detachment_mask_input : np.ndarray, str, or Path
    ventricle_mask_input : np.ndarray, str, or Path, optional
    damage_mask_input : np.ndarray, str, or Path, optional
    full_classification_out_file : str or Path, optional
        Path at which to save pixel classification. Default is None.

    Returns
    -------
    pixel_classification : np.ndarray
        Full pixel classification array
    """
    transcripts_mask = data_processing.process_input(transcripts_mask_input)
    detachment_mask = data_processing.process_input(detachment_mask_input)
    ventricle_mask = data_processing.process_input(ventricle_mask_input) if ventricle_mask_input is not None else None
    damage_mask = data_processing.process_input(damage_mask_input) if damage_mask_input is not None else None

    if ventricle_mask is not None and damage_mask is not None:
        try:
            pixel_classification = (1 * damage_mask +
                                    2 * (transcripts_mask & ~damage_mask) +
                                    3 * (detachment_mask & ~(damage_mask | transcripts_mask)) +
                                    4 * (ventricle_mask & ~(damage_mask | transcripts_mask | detachment_mask)))
        except Exception as e:
            raise Exception(f"Cannot classify pixels on masks of unequal size. "
                            f"Call resize_all_masks first.\n{e}")
    elif ventricle_mask is None and damage_mask is None:
        pixel_classification = (2 * transcripts_mask +
                                3 * (detachment_mask & ~transcripts_mask))
    else:
        raise ValueError("Both or neither of ventricle_mask_input and damage_mask_input must be provided.")

    if full_classification_out_file is not None:
        tiff.imwrite(full_classification_out_file, pixel_classification)

    return pixel_classification


def calculate_class_areas(pixel_classification: np.ndarray) -> tuple:
    """
    Calculates class areas in microns squared

    Parameters
    ----------
    pixel_classification : np.ndarray
        Pixel classification array with values in [0, 1, 2, 3, 4]

    Returns
    -------
    tuple of float
        damage_area, transcripts_area, detachment_area, ventricle_area, total_area (all in µm²)
    """
    damage_area = float(np.sum(pixel_classification == 1) * 100)
    transcripts_area = float(np.sum(pixel_classification == 2) * 100)
    detachment_area = float(np.sum(pixel_classification == 3) * 100)
    ventricle_area = float(np.sum(pixel_classification == 4) * 100)
    total_area = damage_area + transcripts_area + detachment_area + ventricle_area

    return damage_area, transcripts_area, detachment_area, ventricle_area, total_area


def calculate_class_percentages(damage_area: float, transcripts_area: float,
                                 detachment_area: float, ventricle_area: float,
                                 total_area: float) -> tuple:
    """
    Computes percentage of "ideal" tissue area for each pixel class

    Parameters
    ----------
    damage_area : float
    transcripts_area : float
    detachment_area : float
    ventricle_area : float
    total_area : float

    Returns
    -------
    tuple of float
        damage_percent, transcripts_percent, detachment_percent, ventricle_percent
    """
    damage_percent = float(np.round((damage_area / total_area) * 100, 4))
    transcripts_percent = float(np.round((transcripts_area / total_area) * 100, 4))
    detachment_percent = float(np.round((detachment_area / total_area) * 100, 4))
    ventricle_percent = float(np.round((ventricle_area / total_area) * 100, 4))

    return damage_percent, transcripts_percent, detachment_percent, ventricle_percent
