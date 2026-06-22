import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
from typing import Union
import xenquaco.data_processing as data_processing


def transcripts_overview(transcripts: pd.DataFrame,
                          subsample: int = 1000,
                          ax: plt.Axes = None,
                          out_file: Union[str, Path] = '',
                          ms: float = 1,
                          alpha: float = 0.5,
                          color: str = "black",
                          title: str = ''):
    """
    Plots transcripts overview, subsampling 0.1% by default

    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcripts table (must include x_location, y_location columns)
    subsample : int, optional
        Denominator for subsampling transcripts. Default is 1000.
    ax : plt.Axes, optional
        Axes on which to plot. Default is None.
    out_file : str or Path, optional
        Path at which to save plot. Default is ''.
    ms : float, optional
        Marker size. Default is 1.
    alpha : float, optional
        Alpha parameter. Default is 0.5.
    color : str, optional
        Color for plotting. Default is 'black'.
    title : str, optional
        Title for plot. Default is ''.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    x = np.asarray(transcripts['x_location'])
    y = np.asarray(transcripts['y_location'])

    xmax = np.max(x) + 200
    ymax = np.max(y) + 200
    xmin = np.min(x) - 200
    ymin = np.min(y) - 200

    sample_number = int(len(transcripts) / subsample)
    samples = np.random.choice(transcripts.shape[0], sample_number, replace=False)

    ax.plot(x[samples], y[samples], '.', ms=ms, alpha=alpha, color=color)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis('off')

    if title != '':
        ax.set_title(title)

    if out_file != '':
        plt.savefig(out_file)


def transcripts_fov_histogram(transcripts: pd.DataFrame,
                               ax: plt.Axes = None,
                               out_file: Union[str, Path] = '',
                               title: str = ''):
    """
    Creates histogram of transcript counts per FOV

    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcript dataframe (must include fov_name column)
    ax : plt.Axes, optional
        Axes on which to plot. Default is None.
    out_file : str or Path, optional
        Path at which to save figure. Default is ''.
    title : str, optional
        Plot title. Default is ''.

    Returns
    -------
    ax : plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ts_per_fov = transcripts.groupby('fov_name').size()
    logbins = np.geomspace(ts_per_fov.min(), ts_per_fov.max(), 11)
    ax.hist(ts_per_fov, bins=logbins, edgecolor='k', color='slateblue')
    ax.set_xscale('log')
    ax.set_xlabel('Transcript counts per FOV')
    ax.set_ylabel('Frequency (#FOVs)')

    if title != '':
        ax.set_title(title)

    if out_file != '':
        plt.savefig(out_file, dpi=200)

    return ax


def plot_pixel_percentages(transcripts_percent: float,
                            detachment_percent: float,
                            damage_percent: float = np.nan,
                            ventricle_percent: float = np.nan,
                            colormap_list: list = ["white", "orange", "green", "red", "blue"],
                            colormap_labels: list = ["Off-tissue", "Damage", "Tissue", "Detachment", "Ventricles"],
                            ax: Axes = None,
                            title: str = '',
                            out_file: Union[str, Path] = '',
                            dpi: int = 100):
    """
    Plots proportion of 'ideal tissue area' for each classified pixel category

    Parameters
    ----------
    transcripts_percent : float
    detachment_percent : float
    damage_percent : float, optional
    ventricle_percent : float, optional
    colormap_list : list, optional
    colormap_labels : list, optional
    ax : Axes, optional
    title : str, optional
    out_file : str or Path, optional
    dpi : int, optional
    """
    if damage_percent != np.nan and ventricle_percent != np.nan:
        barlist = ax.bar([0, 1, 2, 3],
                         [damage_percent, transcripts_percent, detachment_percent, ventricle_percent])
        ax.set_title("Pixel Percentages of Ideal Tissue Area")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(colormap_labels[1:])
        barlist[0].set_color(colormap_list[1])
        barlist[1].set_color(colormap_list[2])
        barlist[2].set_color(colormap_list[3])
        barlist[3].set_color(colormap_list[4])
        for index, value in enumerate([damage_percent, transcripts_percent,
                                        detachment_percent, ventricle_percent]):
            ax.text(index - 0.15, value + 0.5, f"{str(value)}%")
    elif damage_percent == np.nan and ventricle_percent == np.nan:
        barlist = ax.bar([0, 1], [transcripts_percent, detachment_percent])
        ax.set_title("Pixel Percentages of Ideal Tissue Area")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(colormap_labels[2:4])
        barlist[0].set_color(colormap_list[2])
        barlist[1].set_color(colormap_list[3])
        for index, value in enumerate([transcripts_percent, detachment_percent]):
            ax.text(index - 0.15, value + 0.5, f"{str(value)}%")
    else:
        raise ValueError("Both or neither of damage_percent and ventricle_percent must be provided.")

    if title != '':
        ax.set_title(title)

    if out_file != '':
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight', facecolor='white', transparent=False)


def plot_pixel_classification(pixel_classification: np.ndarray,
                               ax: Axes = None,
                               title: str = '',
                               colormap_list: list = ["white", "orange", "green", "red", "blue"],
                               colormap_labels: list = ["Off-tissue", "Damage", "Tissue", "Detachment", "Ventricles"],
                               out_file: Union[str, Path] = '',
                               dpi: int = 200):
    """
    Plots pixel classification of an experiment

    Parameters
    ----------
    pixel_classification : np.ndarray
        Array of pixel classification results (values 0-4)
    ax : Axes, optional
    title : str, optional
    colormap_list : list, optional
    colormap_labels : list, optional
    out_file : str or Path, optional
    dpi : int, optional
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    cmap = ListedColormap(colormap_list)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N, clip=False)

    img = ax.imshow(pixel_classification, cmap=cmap, norm=norm, interpolation="none")
    cbar = plt.colorbar(img, ticks=[0, 1, 2, 3, 4], ax=ax)
    cbar.set_ticklabels(colormap_labels)

    if title != '':
        ax.set_title(title)

    if out_file != '':
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight', facecolor='white', transparent=False)


def plot_mask(mask_input: Union[np.ndarray, str, Path],
              ax: plt.Axes = None,
              title: str = '',
              out_file: Union[str, Path] = '',
              dpi: int = 200):
    """
    Plots binary mask

    Parameters
    ----------
    mask_input : np.ndarray, str, or Path
    ax : plt.Axes, optional
    title : str, optional
    out_file : str or Path, optional
    dpi : int, optional
    """
    mask = data_processing.process_input(mask_input)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(mask)
    ax.axis('off')
    if title != '':
        ax.set_title(title)

    if out_file != '':
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight', facecolor='white', transparent=False)


def plot_masks(dapi_mask_input: Union[np.ndarray, str, Path],
               dapi_mask_ax: Axes,
               transcripts_mask_input: Union[np.ndarray, str, Path],
               transcripts_mask_ax: Axes,
               detachment_mask_input: Union[np.ndarray, str, Path],
               detachment_mask_ax: Axes,
               damage_mask_input: Union[np.ndarray, str, Path] = None,
               damage_mask_ax: Axes = None,
               ventricle_mask_input: Union[np.ndarray, str, Path] = None,
               ventricle_mask_ax: Axes = None,
               out_file: Union[str, Path] = '',
               dpi: int = 200):
    """
    Plots all binary masks on their respective axes
    """
    dapi_mask = data_processing.process_input(dapi_mask_input)
    transcripts_mask = data_processing.process_input(transcripts_mask_input)
    detachment_mask = data_processing.process_input(detachment_mask_input)
    damage_mask = data_processing.process_input(damage_mask_input) if damage_mask_input is not None else None
    ventricle_mask = data_processing.process_input(ventricle_mask_input) if ventricle_mask_input is not None else None

    if dapi_mask_ax is not None:
        dapi_mask_ax.imshow(dapi_mask)
        dapi_mask_ax.set_title("DAPI Mask")
        dapi_mask_ax.axis('off')

    if transcripts_mask_ax is not None:
        transcripts_mask_ax.imshow(transcripts_mask)
        transcripts_mask_ax.set_title("Transcript Mask")
        transcripts_mask_ax.axis('off')

    if detachment_mask_ax is not None:
        detachment_mask_ax.imshow(detachment_mask)
        detachment_mask_ax.set_title("Detachment Mask")
        detachment_mask_ax.axis('off')

    if damage_mask_ax is not None and damage_mask is not None:
        damage_mask_ax.imshow(damage_mask)
        damage_mask_ax.set_title("Damage Mask")
        damage_mask_ax.axis('off')

    if ventricle_mask_ax is not None and ventricle_mask is not None:
        ventricle_mask_ax.imshow(ventricle_mask)
        ventricle_mask_ax.set_title("Ventricles Mask")
        ventricle_mask_ax.axis('off')

    if out_file != '':
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight', facecolor='white', transparent=False)


def plot_full_pixel_fig(pixel_classification: np.ndarray,
                         dapi_mask_input: Union[np.ndarray, str, Path],
                         transcripts_mask_input: Union[np.ndarray, str, Path],
                         detachment_mask_input: Union[np.ndarray, str, Path],
                         transcripts_percent: Union[int, float],
                         detachment_percent: Union[int, float],
                         damage_mask_input: Union[np.ndarray, str, Path] = None,
                         ventricle_mask_input: Union[np.ndarray, str, Path] = None,
                         damage_percent: float = np.nan,
                         ventricle_percent: float = np.nan,
                         out_file: Union[str, Path] = '',
                         dpi: int = 200):
    """
    Plots full pixel classification figure with pixel classification, pixel percentages, and all binary masks

    Parameters
    ----------
    pixel_classification : np.ndarray
    dapi_mask_input : np.ndarray, str, or Path
    transcripts_mask_input : np.ndarray, str, or Path
    detachment_mask_input : np.ndarray, str, or Path
    transcripts_percent : int or float
    detachment_percent : int or float
    damage_mask_input : np.ndarray, str, or Path, optional
    ventricle_mask_input : np.ndarray, str, or Path, optional
    damage_percent : float, optional
    ventricle_percent : float, optional
    out_file : str or Path, optional
    dpi : int, optional
    """
    dapi_mask = data_processing.process_input(dapi_mask_input)
    transcripts_mask = data_processing.process_input(transcripts_mask_input)
    detachment_mask = data_processing.process_input(detachment_mask_input)
    damage_mask = data_processing.process_input(damage_mask_input) if damage_mask_input is not None else None
    ventricle_mask = data_processing.process_input(ventricle_mask_input) if ventricle_mask_input is not None else None

    if damage_mask is not None and ventricle_mask is not None:
        gs = gridspec.GridSpec(6, 10)
        fig = plt.figure(figsize=(20, 12))

        pixel_class_ax = fig.add_subplot(gs[0:4, 0:5])
        pixel_class_ax.axis('off')
        pixel_perc_ax = fig.add_subplot(gs[0:4, 5:])
        dapi_mask_ax = fig.add_subplot(gs[4:, 0:2])
        dapi_mask_ax.axis('off')
        transcripts_mask_ax = fig.add_subplot(gs[4:, 2:4])
        transcripts_mask_ax.axis('off')
        damage_mask_ax = fig.add_subplot(gs[4:, 4:6])
        damage_mask_ax.axis('off')
        detachment_mask_ax = fig.add_subplot(gs[4:, 6:8])
        detachment_mask_ax.axis('off')
        ventricle_mask_ax = fig.add_subplot(gs[4:, 8:10])
        ventricle_mask_ax.axis('off')

    elif damage_mask is None and ventricle_mask is None:
        gs = gridspec.GridSpec(4, 6)
        fig = plt.figure(figsize=(16, 12))

        pixel_class_ax = fig.add_subplot(gs[0:2, 0:3])
        pixel_class_ax.axis('off')
        pixel_perc_ax = fig.add_subplot(gs[0:2, 3:])
        dapi_mask_ax = fig.add_subplot(gs[2:, 0:2])
        dapi_mask_ax.axis('off')
        transcripts_mask_ax = fig.add_subplot(gs[2:, 2:4])
        transcripts_mask_ax.axis('off')
        detachment_mask_ax = fig.add_subplot(gs[2:, 4:6])
        detachment_mask_ax.axis('off')
        damage_mask_ax = None
        ventricle_mask_ax = None

    else:
        raise ValueError("Both or neither of damage_mask_input and ventricle_mask_input must be provided.")

    plot_pixel_classification(pixel_classification, ax=pixel_class_ax)
    plot_pixel_percentages(transcripts_percent, detachment_percent, damage_percent,
                           ventricle_percent, ax=pixel_perc_ax)
    plot_masks(dapi_mask, dapi_mask_ax, transcripts_mask, transcripts_mask_ax,
               detachment_mask, detachment_mask_ax, damage_mask, damage_mask_ax,
               ventricle_mask, ventricle_mask_ax, out_file)

    fig.subplots_adjust(hspace=0.7)
    plt.suptitle("Pixel Classification", fontsize=20)

    if out_file != '':
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight', facecolor='white', transparent=False)

    plt.show()
    plt.close()
