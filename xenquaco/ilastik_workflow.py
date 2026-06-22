from typing import Union
from pathlib import Path
import subprocess


def get_pixel_workflow_args(ilastik_program_path: Union[str, Path],
                            pixel_classification_model_path: Union[str, Path],
                            input_image_path: Union[str, Path],
                            probability_map_path: Union[str, Path],
                            export_type: str = 'Probabilities') -> list:
    """
    Helper function creates subprocess arguments for pixel classification workflow

    Parameters
    ----------
    ilastik_program_path : str or Path
        Path to ilastik program
    pixel_classification_model_path : str or Path
        Path to pixel classification model
    input_image_path : str or Path
        Path to input image on which to perform pixel classification
    probability_map_path : str or Path
        Path at which to save probability map
    export_type : str, optional
        Export type for pixel classification. Default is 'Probabilities'.

    Returns
    -------
    list
        Arguments to pass to subprocess for running ilastik pixel classification workflow
    """
    return [
        ilastik_program_path,
        '--headless',
        '--project',
        pixel_classification_model_path,
        '--export_source',
        export_type,
        '--raw_data',
        input_image_path,
        '--output_format=tiff',
        '--output_filename_format',
        probability_map_path,
    ]


def get_object_workflow_args(ilastik_program_path: Union[str, Path],
                             object_classification_model_path: Union[str, Path],
                             input_image_path: Union[str, Path],
                             probability_map_path: Union[str, Path],
                             mask_path: Union[str, Path]) -> list:
    """
    Helper function creates subprocess arguments for object classification workflow

    Parameters
    ----------
    ilastik_program_path : str or Path
        Path to ilastik program
    object_classification_model_path : str or Path
        Path to object classification model
    input_image_path : str or Path
        Path to input image
    probability_map_path : str or Path
        Path to probability map file
    mask_path : str or Path
        Path at which to save mask

    Returns
    -------
    list
        Arguments to pass to subprocess for running ilastik object classification workflow
    """
    return [
        ilastik_program_path,
        '--headless',
        '--project',
        object_classification_model_path,
        '--export_source',
        'Object Predictions',
        '--prediction_maps',
        probability_map_path,
        '--raw_data',
        input_image_path,
        '--output_format=tiff',
        '--output_filename_format',
        mask_path,
    ]


def run_ilastik_workflow(workflow_args: list):
    """
    Run ilastik workflow through subprocess

    Parameters
    ----------
    workflow_args : list
        Arguments to pass to ilastik workflow

    Raises
    ------
    RuntimeError
        If ilastik workflow fails to execute successfully
    """
    try:
        result = subprocess.run(workflow_args,
                                stdout=open('ilastik_logs.txt', 'w+'),
                                stderr=open('ilastik_logs.txt', 'w+'))
        if result.returncode != 0:
            raise RuntimeError('Ilastik workflow failed')
    except Exception as e:
        raise RuntimeError(f"Failed to run ilastik workflow: {e}")
