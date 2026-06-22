import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt

import xenquaco.pixel_classification as pc
import xenquaco.data_processing as data_processing
import xenquaco.figures as figures
from xenquaco.__init__ import __version__ as version


# Keys for QC summary JSON output
metrics_dict_keys = [
    "filtered_transcript_count",
    "transcript_density_um2",
    "transcript_density_um2_per_gene",
    "on_tissue_transcript_count",
    "counts_per_gene",
    "damage_area",
    "transcripts_area",
    "detachment_area",
    "ventricle_area",
    "total_area",
    "damage_percent",
    "transcripts_percent",
    "detachment_percent",
    "ventricle_percent",
    "transcripts_mask_pixel_path",
    "transcripts_mask_object_path",
    "dapi_mask_pixel_path",
    "dapi_mask_object_path",
    "ventricle_mask_pixel_path",
    "ventricle_mask_object_path",
    "xenquaco_version",
]


def read_transcripts(transcripts_path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads and returns transcripts table

    Parameters
    ----------
    transcripts_path : str or Path
        Path to transcripts.parquet or transcripts.csv

    Returns
    -------
    transcripts : pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If file not found at given path
    """
    transcripts_path = str(transcripts_path)
    if transcripts_path.endswith('.parquet'):
        try:
            return pd.read_parquet(transcripts_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Transcripts parquet not found at {transcripts_path}: {e}')
    else:
        try:
            return pd.read_csv(transcripts_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Transcripts CSV not found at {transcripts_path}: {e}')


def find_fovs(transcripts: pd.DataFrame) -> pd.DataFrame:
    """
    Groups transcript table by FOV and stores coordinate and count information

    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcripts table (must include x_location, y_location, fov_name columns)

    Returns
    -------
    fovs : pd.DataFrame
        FOVs dataframe with coordinates and transcript counts
    """
    fovs = transcripts[['x_location', 'y_location', 'fov_name']].groupby('fov_name').min()
    fovs.rename(columns={'x_location': 'x_min', 'y_location': 'y_min'}, inplace=True)
    fovs[['x_max', 'y_max']] = transcripts[['x_location', 'y_location', 'fov_name']].groupby('fov_name').max()

    fovs['width'] = fovs['x_max'] - fovs['x_min']
    fovs['height'] = fovs['y_max'] - fovs['y_min']

    fovs['center_x'] = (fovs['x_max'] - fovs['x_min']) / 2 + fovs['x_min']
    fovs['center_y'] = (fovs['y_max'] - fovs['y_min']) / 2 + fovs['y_min']

    fovs['transcript_counts'] = transcripts.groupby('fov_name').size()

    return fovs


def get_fov_neighbors(fovs: pd.DataFrame) -> pd.DataFrame:
    """
    Gets cardinal neighbor FOVs for each FOV using grid coordinates

    Parameters
    ----------
    fovs : pd.DataFrame
        FOVs dataframe

    Returns
    -------
    fovs : pd.DataFrame
        Updated FOVs dataframe with neighbor locations
    """
    max_width = np.max(fovs['width'])
    max_height = np.max(fovs['height'])
    centers_array = np.array(fovs[['center_x', 'center_y']])
    neighbors = [[] for _ in range(len(fovs))]

    for i in range(len(fovs)):
        fov = fovs.index[i]
        fov_center = np.broadcast_to(centers_array[i], (len(centers_array), 2))
        euclidian_distances = np.argsort(np.linalg.norm(fov_center - centers_array, axis=1))

        above_fovs = np.unique(np.where(
            (centers_array[:, 1] > fovs.loc[fov, 'y_max']) &
            (abs(centers_array[:, 0] - centers_array[i, 0]) <= max_height / 2) &
            (abs(centers_array[:, 1] - centers_array[i, 1]) <= max_height * 1.5))[0])
        if len(above_fovs) > 0:
            neighbors[i].append(fovs.index[euclidian_distances[np.isin(euclidian_distances, above_fovs)][0]])

        below_fovs = np.unique(np.where(
            (centers_array[:, 1] < fovs.loc[fov, 'y_min']) &
            (abs(centers_array[:, 0] - centers_array[i, 0]) <= max_height / 2) &
            (abs(centers_array[:, 1] - centers_array[:, 1]) <= max_height * 1.5))[0])
        if len(below_fovs) > 0:
            neighbors[i].append(fovs.index[euclidian_distances[np.isin(euclidian_distances, below_fovs)][0]])

        right_fovs = np.unique(np.where(
            (centers_array[:, 0] > fovs.loc[fov, 'x_max']) &
            (abs(centers_array[:, 1] - centers_array[i, 1]) <= max_width / 2) &
            (abs(centers_array[:, 0] - centers_array[i, 0]) <= max_width * 1.5))[0])
        if len(right_fovs) > 0:
            neighbors[i].append(fovs.index[euclidian_distances[np.isin(euclidian_distances, right_fovs)][0]])

        left_fovs = np.unique(np.where(
            (centers_array[:, 0] < fovs.loc[fov, 'x_min']) &
            (abs(centers_array[:, 1] - centers_array[i, 1]) <= max_width / 2) &
            (abs(centers_array[:, 0] - centers_array[i, 0]) <= max_width * 1.5))[0])
        if len(left_fovs) > 0:
            neighbors[i].append(fovs.index[euclidian_distances[np.isin(euclidian_distances, left_fovs)][0]])

    fovs['neighbors'] = neighbors
    return fovs


def get_fovs_dataframe(transcripts: pd.DataFrame) -> pd.DataFrame:
    """
    Creates FOVs dataframe including coordinates, transcript counts, neighbors, and counts per gene

    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcripts table

    Returns
    -------
    fovs : pd.DataFrame
    """
    fovs = find_fovs(transcripts)
    fovs = get_fov_neighbors(fovs)
    counts_per_gene = transcripts.groupby('fov_name')['feature_name'].value_counts().unstack(fill_value=0)
    fovs = fovs.merge(counts_per_gene, left_index=True, right_index=True, how='left')
    return fovs


def get_transcript_density(transcripts_image_input: Union[np.ndarray, str, Path],
                            transcripts_mask_input: Union[np.ndarray, str, Path]) -> float:
    """
    Calculates transcript density per on-tissue micron squared

    Parameters
    ----------
    transcripts_image_input : np.ndarray, str, or Path
        Array of or path to transcripts image
    transcripts_mask_input : np.ndarray, str, or Path
        Array of or path to binary transcripts mask

    Returns
    -------
    transcript_density_um2 : float
        Number of transcripts per on-tissue µm²
    """
    transcripts_image = data_processing.process_input(transcripts_image_input)
    transcripts_mask = data_processing.process_input(transcripts_mask_input)
    on_tissue_count = get_on_tissue_transcript_count(transcripts_image, transcripts_mask)
    mask_area = np.count_nonzero(transcripts_mask) * 100  # 10µm pixels → 100µm² per pixel

    if mask_area > 0:
        return on_tissue_count / mask_area
    else:
        return np.nan


def get_on_tissue_transcript_count(transcripts_image_input: Union[np.ndarray, str, Path],
                                    transcripts_mask_input: Union[np.ndarray, str, Path]) -> int:
    """
    Calculates number of on-tissue transcripts

    Parameters
    ----------
    transcripts_image_input : np.ndarray, str, or Path
    transcripts_mask_input : np.ndarray, str, or Path

    Returns
    -------
    int
        Number of on-tissue transcripts
    """
    transcripts_image = data_processing.process_input(transcripts_image_input)
    transcripts_mask = data_processing.process_input(transcripts_mask_input)
    return int(np.sum(transcripts_image[transcripts_mask == 1]))


def write_qc_summary(qc_summary_path: Union[str, Path], qc_dict: dict) -> None:
    """
    Writes or updates JSON file with QC metrics

    Parameters
    ----------
    qc_summary_path : str or Path
        Path to output JSON file
    qc_dict : dict
        Dictionary of QC metric keys and values
    """
    if os.path.exists(qc_summary_path):
        with open(qc_summary_path, 'r') as file:
            data = json.load(file)
        for key, value in qc_dict.items():
            if isinstance(data.get(key), float) and np.isnan(data.get(key)):
                data[key] = value
    else:
        data = qc_dict.copy()

    with open(qc_summary_path, 'w') as file:
        json.dump(data, file, indent=4)


class XeniumExperiment:

    def __init__(self,
                 transcripts_input: Union[pd.DataFrame, str, Path],
                 ilastik_program_path: Union[str, Path] = None,
                 dapi_high_res_image_path: Union[str, Path] = None,
                 output_dir: Union[str, Path] = None,
                 ventricle_genes_list: list = ["Crb2", "Glis3", "Inhbb", "Naaa", "Cd24a",
                                               "Dsg2", "Hdc", "Shroom3", "Vit", "Rgs12", "Trp73"],
                 force_mask: bool = False):
        """
        Initialize a XeniumExperiment from a transcripts table

        Parameters
        ----------
        transcripts_input : pd.DataFrame, str, or Path
            DataFrame of or path to transcripts.parquet (or transcripts.csv)
        ilastik_program_path : str or Path, optional
            Path to ilastik executable. Default is None.
        dapi_high_res_image_path : str or Path, optional
            Path to high-resolution DAPI image (e.g. morphology_focus.ome.tif). Default is None.
        output_dir : str or Path, optional
            Directory at which to save QC outputs. Default is ./qc_output.
        ventricle_genes_list : list, optional
            List of ventricle marker genes. Default is mouse ventricle markers.
        force_mask : bool, optional
            Whether to regenerate transcript mask even if it already exists. Default is False.

        Attributes Set
        --------------
        transcripts : pd.DataFrame
            Full transcripts table with Xenium-native column names
        filtered_transcripts : pd.DataFrame
            Transcripts filtered by QV >= 20 and excluding control codewords
        filtered_transcript_count : int
            Number of filtered transcripts
        total_transcript_count : int
            Number of total transcripts (pre-filtering)
        n_genes : int
            Number of unique genes
        genes : list
            List of unique gene names
        counts_per_gene : dict
            Transcript counts per gene (filtered)
        fovs_df : pd.DataFrame
            FOV-level dataframe with coordinates, counts, and neighbors
        """
        if output_dir is None:
            output_dir = Path(os.getcwd(), 'qc_output')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Output paths for masks and images
        self.transcripts_image_path = Path(output_dir, 'transcripts.tiff')
        self.transcripts_mask_path = Path(output_dir, 'transcripts_mask.tiff')
        self.dapi_image_path = Path(output_dir, 'dapi.tiff')
        self.dapi_mask_path = Path(output_dir, 'dapi_mask.tiff')
        self.detachment_mask_path = Path(output_dir, 'detachment_mask.tiff')
        self.ventricle_image_path = Path(output_dir, 'ventricles.tiff')
        self.ventricle_mask_path = Path(output_dir, 'ventricles_mask.tiff')
        self.damage_mask_path = Path(output_dir, 'damage_mask.tiff')
        self.pixel_classification_path = Path(output_dir, 'pixel_classification.tiff')

        self.ilastik_program_path = ilastik_program_path
        self.dapi_high_res_image_path = dapi_high_res_image_path
        self.output_dir = output_dir
        self.ventricle_genes_list = ventricle_genes_list

        # Paths to bundled ilastik models
        ilastik_models_dir = os.path.join(os.path.dirname(__file__), '..', 'ilastik_models')
        self.transcripts_mask_pixel_path = os.path.normpath(
            Path(ilastik_models_dir, 'TissueMaskPixelClassification_v1.0.ilp'))
        self.transcripts_mask_object_path = os.path.normpath(
            Path(ilastik_models_dir, 'TissueMaskObjects_v1.1.ilp'))
        self.dapi_mask_pixel_path = os.path.normpath(
            Path(ilastik_models_dir, 'DapiMaskPixelClassification_Mouse.ilp'))
        self.dapi_mask_object_path = os.path.normpath(
            Path(ilastik_models_dir, 'DapiMaskObjectClassification_Mouse.ilp'))
        self.ventricle_mask_pixel_path = os.path.normpath(
            Path(ilastik_models_dir, 'VentriclesPixelClassification.ilp'))
        self.ventricle_mask_object_path = os.path.normpath(
            Path(ilastik_models_dir, 'VentriclesObjectClassification.ilp'))

        # Process transcripts — keep Xenium-native column names throughout
        print('Processing transcripts dataframe')
        self.transcripts = data_processing.process_input(transcripts_input)

        # Filter by quality score and remove control codewords
        print('Filtering transcripts')
        filtered = self.remove_low_quality_transcripts(self.transcripts)
        self.filtered_transcripts = self.remove_controls(filtered)

        # Basic experiment metadata
        self.n_genes = self.filtered_transcripts['feature_name'].nunique()
        self.genes = list(self.filtered_transcripts['feature_name'].unique())
        self.counts_per_gene = self.filtered_transcripts.groupby('feature_name').size().to_dict()
        self.total_transcript_count = len(self.transcripts)
        self.filtered_transcript_count = len(self.filtered_transcripts)

        # FOV dataframe
        print('Creating FOVs dataframe')
        self.fovs_df = get_fovs_dataframe(self.filtered_transcripts)

        # Package version
        self.xenquaco_version = version

        # Generate transcript mask if ilastik path provided
        self.transcripts_mask = None
        if self.ilastik_program_path is not None:
            if not os.path.exists(self.transcripts_mask_path) or force_mask:
                print('Generating transcripts mask')
                self.transcripts_mask = pc.generate_transcripts_mask(
                    self.transcripts_image_path,
                    self.ilastik_program_path,
                    self.transcripts_mask_pixel_path,
                    self.transcripts_mask_object_path,
                    self.filtered_transcripts)
            else:
                print('Reading existing transcripts mask')
                self.transcripts_mask = data_processing.process_path(str(self.transcripts_mask_path))

    @staticmethod
    def remove_low_quality_transcripts(transcripts: pd.DataFrame, val: int = 20) -> pd.DataFrame:
        """
        Filters transcripts by Quality Value (QV) score

        Parameters
        ----------
        transcripts : pd.DataFrame
            Transcripts table (must include qv column)
        val : int, optional
            Minimum QV score. Default is 20.

        Returns
        -------
        pd.DataFrame
            Transcripts with qv >= val
        """
        if 'qv' not in transcripts.columns:
            raise KeyError('transcripts dataframe must include "qv" column')
        return transcripts[transcripts['qv'] >= val]

    @staticmethod
    def remove_controls(transcripts: pd.DataFrame, which: str = 'all') -> pd.DataFrame:
        """
        Removes Xenium control codewords from transcripts table

        Parameters
        ----------
        transcripts : pd.DataFrame
            Transcripts table (must include feature_name column)
        which : str, optional
            Which controls to remove. One of ['all', 'NegControlCodeword', 'NegControlProbe',
            'UnassignedCodeword', 'DeprecatedCodeword']. Default is 'all'.

        Returns
        -------
        pd.DataFrame
            Transcripts excluding specified control codewords
        """
        valid_which = ['all', 'NegControlCodeword', 'NegControlProbe',
                       'UnassignedCodeword', 'DeprecatedCodeword']
        if which not in valid_which:
            raise KeyError(f'`which` must be one of {valid_which}')

        if which == 'all':
            return transcripts[~transcripts['feature_name'].str.startswith(
                ('NegControl', 'Unassigned', 'Deprecated'))]
        else:
            return transcripts[~transcripts['feature_name'].str.startswith(which)]

    def run_full_pixel_classification(self, save_metrics: bool = True):
        """
        Runs entire pixel classification workflow:
            - Generates binary masks for transcripts, DAPI, detachment, ventricles, and damage
            - Resizes and aligns masks
            - Calculates pixel areas and percentages of ideal tissue area

        Parameters
        ----------
        save_metrics : bool, optional
            Whether to save pixel stats to pixel_stats.json. Default is True.

        Attributes Set
        --------------
        transcripts_mask, dapi_mask, detachment_mask, ventricle_mask, damage_mask : np.ndarray
        pixel_classification : np.ndarray
        damage_area, transcripts_area, detachment_area, ventricle_area, total_area : float
        damage_percent, transcripts_percent, detachment_percent, ventricle_percent : float
        """
        self.ventricle_mask = None
        self.damage_mask = None

        if self.transcripts_mask is None:
            print("Generating transcripts mask...")
            self.transcripts_mask = pc.generate_transcripts_mask(
                self.transcripts_image_path,
                self.ilastik_program_path,
                self.transcripts_mask_pixel_path,
                self.transcripts_mask_object_path,
                self.filtered_transcripts)

        print("Generating DAPI mask...")
        self.dapi_mask = pc.generate_dapi_mask(
            self.dapi_image_path,
            self.ilastik_program_path,
            self.dapi_mask_pixel_path,
            self.dapi_mask_object_path,
            self.dapi_high_res_image_path)

        print("Generating detachment mask...")
        self.detachment_mask = pc.generate_detachment_mask(
            self.transcripts_mask_path,
            self.dapi_mask_path,
            self.detachment_mask_path)

        if any(np.isin(self.genes, self.ventricle_genes_list)):
            print("Generating ventricle mask...")
            self.ventricle_mask = pc.generate_ventricle_mask(
                self.ventricle_image_path,
                self.dapi_mask_path,
                self.transcripts_mask_path,
                self.ilastik_program_path,
                self.ventricle_mask_pixel_path,
                self.ventricle_mask_object_path,
                self.filtered_transcripts,
                self.ventricle_genes_list)

            print("Generating damage mask...")
            self.damage_mask = pc.generate_damage_mask(
                self.damage_mask_path,
                self.dapi_image_path,
                self.dapi_mask_path,
                self.transcripts_mask_path,
                self.ventricle_mask_path)

            self.transcripts_mask, self.dapi_mask, self.detachment_mask, \
                self.ventricle_mask, self.damage_mask = pc.resize_all_masks(
                    self.transcripts_mask, self.dapi_mask, self.detachment_mask,
                    self.ventricle_mask, self.damage_mask)

        print("Classifying pixels...")
        self.pixel_classification = pc.classify_pixels(
            self.transcripts_mask,
            self.detachment_mask,
            self.ventricle_mask,
            self.damage_mask,
            self.pixel_classification_path)

        self.damage_area, self.transcripts_area, self.detachment_area, \
            self.ventricle_area, self.total_area = pc.calculate_class_areas(self.pixel_classification)
        self.damage_percent, self.transcripts_percent, self.detachment_percent, \
            self.ventricle_percent = pc.calculate_class_percentages(
                self.damage_area, self.transcripts_area, self.detachment_area,
                self.ventricle_area, self.total_area)

        if self.output_dir is not None and save_metrics:
            pixel_stats_dict = {
                'damage_area': self.damage_area,
                'transcripts_area': self.transcripts_area,
                'detachment_area': self.detachment_area,
                'ventricle_area': self.ventricle_area,
                'damage_percent': self.damage_percent,
                'transcripts_percent': self.transcripts_percent,
                'detachment_percent': self.detachment_percent,
                'ventricle_percent': self.ventricle_percent,
                'total_area': self.total_area,
            }
            with open(Path(self.output_dir, 'pixel_stats.json'), 'w') as f:
                json.dump(pixel_stats_dict, f, indent=4)

    def run_all_qc(self,
                   run_pixel_classification: bool = True,
                   plot_figures: bool = True,
                   save_metrics: bool = True):
        """
        Runs all standard Xenium QC functions

        Parameters
        ----------
        run_pixel_classification : bool, optional
            Whether to run full pixel classification workflow. Default is True.
        plot_figures : bool, optional
            Whether to generate and save figures. Default is True.
        save_metrics : bool, optional
            Whether to save QC summary JSON. Default is True.
        """
        # 1. Pixel classification
        if run_pixel_classification:
            self.run_full_pixel_classification(save_metrics)

            if plot_figures:
                figures.plot_full_pixel_fig(
                    self.pixel_classification,
                    self.dapi_mask,
                    self.transcripts_mask,
                    self.detachment_mask,
                    self.transcripts_percent,
                    self.detachment_percent,
                    self.damage_mask,
                    self.ventricle_mask,
                    self.damage_percent,
                    self.ventricle_percent,
                    Path(self.output_dir, 'pixel_classification_plot.png'))
                plt.show()

        # 2. On-tissue transcript metrics
        self.on_tissue_transcript_count = get_on_tissue_transcript_count(
            self.transcripts_image_path, self.transcripts_mask)
        self.transcript_density_um2 = get_transcript_density(
            self.transcripts_image_path, self.transcripts_mask)
        self.transcript_density_um2_per_gene = self.transcript_density_um2 / self.n_genes

        if plot_figures:
            figures.transcripts_overview(
                self.filtered_transcripts,
                out_file=Path(self.output_dir, 'transcripts_overview.png'))

        # 3. Save QC summary
        metrics_dict = {key: getattr(self, key, np.nan) for key in metrics_dict_keys}

        if save_metrics:
            write_qc_summary(Path(self.output_dir, 'qc_summary.json'), metrics_dict)
