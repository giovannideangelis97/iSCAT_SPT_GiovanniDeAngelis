import numpy as np
import imgrvt as rvt
import os
import trackpy as tp
import matplotlib.pyplot as plt
import argparse
import json
import pims
import warnings
import tifffile as tiff
import math
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from queue import Queue
from tqdm import tqdm
from typing import Union
from dataclasses import dataclass
from multiprocessing import Process, Manager

@dataclass
class RVTSettings:
    min_radius : int = 2
    max_radius : int = 25
    rvt_kind : str = "normalized"
    highpass : int = 5
    coarse_factor : int = 1
    coarse_mode : str = "add"
    pad_mode : str = "constant"


@dataclass
class TrackSettings:
    locate : dict
    link : dict
    annotate : dict

    def __post_init__(self):
        diameter = int(round(self.locate["locate_half_width_gaussian"] / self.locate["locate_micron_per_pixel"]))*2 + 1
        self.locate["locate_true_radius"] = diameter

def get_file_list(folder: str, file_type: str) -> list:
    """Returns a list with all the files with a specific file extension.

    Args:
        folder (str): folder to look into
        file_type (str): extension of the file type to look into
    
    Returns:
        list of str with all filenames found with specific extension.
    """
    file_list = []
    for path, _, files in os.walk(folder):
            for name in files:
                if name.endswith(file_type) and not "_RVT" in name:
                    file_list.append(os.path.join(path, name))
    return file_list

@pims.pipeline
def apply_rvt(input_frame: np.ndarray,
            rvt_settings: RVTSettings) -> np.ndarray:

        """ Converts the input frame to float32 and applies
        Radial Variance Transform.
        """
           
        input_frame = input_frame.astype(np.float32)
        frame_rvt = rvt.rvt(input_frame, 
                    rvt_settings.min_radius, 
                    rvt_settings.max_radius, 
                    rvt_settings.rvt_kind, 
                    highpass_size=rvt_settings.highpass, 
                    coarse_factor=rvt_settings.coarse_factor,
                    coarse_mode=rvt_settings.coarse_mode,
                    pad_mode=rvt_settings.pad_mode)
        return frame_rvt

def locate_single_frame(video: np.ndarray,
                    track_settings: TrackSettings):
    """ Calls `trackpy.locate` on a single frame with the specified tracking parameters.
    """

    idx = track_settings.locate["locate_frame_index"]
    frame = video[idx]

    mean = np.mean(frame)
    thr_perc = track_settings.locate["locate_value_threshold"]
    
    threshold = np.max(frame)*thr_perc

    print(f"Average pixel value: {mean}")
    print(f"Threshold value: {threshold} ({thr_perc*100} %)")

    try:
        df = tp.locate(frame,
                        invert=track_settings.locate["locate_invert_frame"],
                        diameter=track_settings.locate["locate_true_radius"],
                        minmass=track_settings.locate["locate_minmass"],
                        maxsize=track_settings.locate["locate_maxsize"],
                        threshold=threshold)
        tp.annotate(df, frame)

        _, ax = plt.subplots()
        ax.hist(df['mass'], bins=20)
        ax.set(xlabel='mass', ylabel='count')
        tp.subpx_bias(df)
        plt.show()
    except:
        print(f"No particles found on frame {idx}")

def annotate_frame(frame: np.ndarray, 
                    settings: dict,
                    plot_style: dict,
                    figure_style: dict, 
                    queue: Queue):


    with plt.ioff(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = tp.locate(frame,
                            invert=settings["locate_invert_frame"],
                            diameter=settings["locate_true_radius"], 
                            minmass=settings["locate_minmass"], 
                            maxsize=settings["locate_maxsize"],
                            threshold=settings["locate_value_threshold"]*np.mean(frame))
        fig, ax = plt.subplots(**figure_style)
        tp.annotate(features, frame, ax=ax, plot_style=plot_style)
        ax.set(yticks=[], xticks=[])
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        plt.close("all")
        marked_video = np.array(fig.canvas.buffer_rgba())
        marked_video = np.rint(np.dot(marked_video[..., :3], [0.2126, 0.7152, 0.0722]))
        queue.put((marked_video, features))

def write_to_file(video_file: str, 
                pandas_file: str,
                storage_class: Union[tp.PandasHDFStore, tp.PandasHDFStoreBig], 
                queue: Queue):
    with tiff.TiffWriter(video_file) as video, storage_class(pandas_file, mode="w") as dataframe:
        while True:
            data = queue.get()
            if data is not None:
                video.write(data[0], contiguous=True)
                dataframe.put(data[1])
            else:
                break


def locate_tracks(video: np.ndarray,
                outdf: str,
                outvideo: str,
                track_settings: TrackSettings,
                plotting : str = None,
                verbose : bool = False):
    """ Estimates the presence of particles within a recorded video and shows a plot with all the identified tracks.
    The behavior depends on wether only the localization on the head frame is requested or not.
    If `locate_only` is set to True, the function will:

    - call `trackpy.locate` on the head frame of the recording using the input track settings;
    - annotate on the frame every particle found in respect to the input settings;
    - plot an instogram of the computed masses of the detected particles;
    - plot an instogram of the computed subpixel bias to check if x-y positions are evenly distributed.

    Args:
        video (numpy.ndarray): input video recording
        track_settings (TrackSettings): dataclass containing parameters needed by the trackpy library for particle detection and track linking.
        plotting (str): creates a plot of the tracking data\\; 
                        can be None to skip plot, "traj" for tracjectory, "subpx" for subpixel precision or "all" for both (default is None)
        verbose (bool): enables trackpy full log visualization (default is False)
    """

    tp.quiet(suppress=not verbose)

    manager = Manager()
    queue = manager.Queue()

    # Depending on the video length we select a different storage method.
    # This is reccomended by the trackpy library.
    if len(video) <= 100:
        storage_class = tp.PandasHDFStore
    else:
        storage_class = tp.PandasHDFStoreBig
    
    locate_outfile = outdf.replace(".h5", "_positions.h5")
    link_outfile = outdf.replace(".h5", "_track_coordinates.h5")

    print(f"Output HDF5 for particle locations: {locate_outfile}")
    print(f"Output HDF5 for particle tracks: {link_outfile}")

    plot_style = dict(
        markersize = track_settings.annotate["markersize"],
        markeredgewidth = track_settings.annotate["markeredgewidth"],
        markerfacecolor = track_settings.annotate["markerfacecolor"],
        markeredgecolor = track_settings.annotate["markeredgecolor"],
        marker = track_settings.annotate["marker"]
    )

    dpi = track_settings.annotate["dpi"]

    try:
        figure_style = dict(
            frameon = False,
            figsize=(video.frame_shape[1]/dpi, video.frame_shape[0]/dpi),
            dpi = dpi
        )
    except AttributeError:
        figure_style = dict(
            frameon = False,
            figsize=(video.shape[2]/dpi, video.shape[1]/dpi),
            dpi = dpi
        )

    # locate the particles in each frame
    write_process = Process(target=write_to_file, args=(outvideo, locate_outfile, storage_class, queue))
    write_process.start()
    with tqdm_joblib(tqdm(desc="trackpy.locate", total=len(video))):
        Parallel(n_jobs=-2)(delayed(annotate_frame)(frame, track_settings.locate, plot_style, figure_style, queue) for frame in video)
    queue.put(None)
    write_process.join()

    # finally: link the particles
    with storage_class(link_outfile, mode="w") as link_s:
        with storage_class(locate_outfile, mode="r") as loc_s:
            try:
                for linked in tqdm(tp.link_df_iter(loc_s,
                                                search_range=track_settings.link["link_search_range"],
                                                memory=track_settings.link["link_frame_memory"],
                                                link_strategy=track_settings.link["link_strategy"]), desc="trackpy.link_df_iter"):
                    link_s.put(linked)
            except Exception as exc:
                print(f"No tracks were found. Adjust the track settings accordingly. ({exc})")
                return

    # draw the tracks on the video and store it
    if not os.path.isdir("./tracked_videos"):
        os.mkdir("./tracked_videos")
    
    if plotting is not None:
        if plotting == "traj":
            with storage_class(link_outfile, mode="r") as f:
                df = f.dump()
            tp.plot_traj(df, mpp=track_settings.locate["locate_micron_per_pixel"])
        elif plotting == "subpx":
            with storage_class(locate_outfile, mode="r") as f:
                df = f.dump()
            tp.subpx_bias(df)
        else: # plotting == "all"
            with storage_class(link_outfile, mode="r") as f:
                link_df = f.dump()
            with storage_class(locate_outfile, mode="r") as f:
                loc_df = f.dump()
            tp.plot_traj(link_df, mpp=track_settings.locate["locate_micron_per_pixel"])
            tp.subpx_bias(loc_df)

def check_file_extension(parser: argparse.ArgumentParser, supported_extensions: list, filename: str) -> str:
    """ Checks wether the input string has a supported file extension.

    Args:
        parser (argparse.ArgumentParser): reference to the main argument parser object.
        supported_extensions (list[str]): list containing the supported file extensions as strings objects.
        filename (str): input file name to check.

    Returns:
        str: the validated filename if extension is supported; otherwise parser throws error.
    """
    ext = os.path.splitext(filename)[1][1:] 
    if ext not in supported_extensions:
        parser.error(f"file extension \"{ext}\" not supported, only TIFF extensions supported")
    return filename

def check_plot_command(parser: argparse.ArgumentParser, supported_plots: list, plot_name: str) -> str:
    if plot_name not in supported_plots:
        parser.error(f"plot command \"{plot_name}\" unrecognized, supported commands: {supported_plots} ")
    return plot_name


if __name__ == "__main__":

    __description__ = """
    iScat particle tracking for video post-processing.
    This script takes in input a recorded video (in TIFF format)
    and uses the trackpy package to detect any particle objects
    within the video frames. Optionally, this script can also
    apply the Radial Variance Transform (RVT) on the input video.
    The input settings for trackpy and RVT are provided via an
    input JSON file.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__description__
    )
    parser.add_argument(
        "video_path", 
        help="absolute or relative path to the input video (only TIFF extension supported)", 
        type=lambda filename : check_file_extension(parser, ["tif","tiff"], filename)
    )
    parser.add_argument(
        "settings_path", 
        help="absolute or relative path to the input JSON settings file", 
        type=str
    )
    parser.add_argument(
        "-odf", "--output_dataframe",
        help="absolute or relative path to the output HDF5 dataframe file",
        type=str
    )
    parser.add_argument(
        "-ovd", "--output_video",
        help="absolute or relative path to the annotated TIFF video",
        type=str
    )
    parser.add_argument(
        "-v", "--verbose", 
        help="disables trackpy.quiet setting (enabled by default)", 
        action="store_true"
    )
    parser.add_argument(
        "-rvt" ,"--apply-rvt", 
        help="apply Radial Variance Transform (RVT) to input video", 
        action="store_true"
    )
    parser.add_argument(
        "-loc", "--locate",
        help="only calls \"trackpy.locate\" on the first video frame",
        action="store_true"
    )
    parser.add_argument(
        "-plt", "--plotting",
        help="plots the obtained data depending on the type of requested plot",
        type=lambda command: check_plot_command(parser, ["traj", "subpx", "all"], command)
    )
    args = parser.parse_args()

    print("Parsing JSON settings file... ", end="")
    with open(args.settings_path, "r") as file:
        settings = json.loads(file.read())
    print("done!")

    print("Creating internal settings... ", end="")

    track_settings = TrackSettings(
        locate=settings["Tracking"]["locate"],
        link=settings["Tracking"]["link"],
        annotate=settings["Tracking"]["annotate"]
    )
    print("done!")

    rvt_settings = None

    if args.apply_rvt == True:
        print("RVT requested - creating settings... ", end="")
        rvt_settings = RVTSettings(
            min_radius = settings["RVT"]["min_radius"],
            max_radius = settings["RVT"]["max_radius"],
            rvt_kind = settings["RVT"]["rvt_kind"],
            highpass = settings["RVT"]["highpass"],
            coarse_factor = settings["RVT"]["coarse_factor"],
            coarse_mode = settings["RVT"]["coarse_mode"],
            pad_mode = settings["RVT"]["pad_mode"],
        )
        print("done!")
    
    def abs_pixels(input: np.ndarray) -> np.ndarray:
        return np.abs(np.around(input)).astype(np.uint16)

    # Read input file
    # if RVT flag is set,
    # call the pims pipeline function

    if track_settings.locate["locate_invert_pixels"]:
        video = abs_pixels(pims.open(args.video_path))
    elif args.apply_rvt:
        video = apply_rvt(pims.open(args.video_path), rvt_settings)
    else:
        video = pims.open(args.video_path)
    
    if args.output_dataframe is None or args.output_video is None:
        if not os.path.isdir("./datasets"):
            os.mkdir("./datasets")

    out_df : str = args.output_dataframe
    if args.output_dataframe is None:
        out_df = "./datasets/particle_data.h5"
    else:
        if not out_df.endswith(".h5") and not out_df.endswith(".hdf5"):
            out_df = out_df + ".h5"
    
    if args.output_video is None:
        out_video = "./datasets/annotated_video.tiff"


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.locate:
            locate_single_frame(video,
                                track_settings)
        else:
            locate_tracks(video,
                            out_df,
                            out_video,
                            track_settings,
                            args.plotting,
                            args.verbose)