import itertools
import os
import pickle
import shutil
import subprocess
import time
from typing import Callable, Optional

import modal
import numpy as np
import psutil
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydantic import BaseModel
from tqdm import tqdm

stub = modal.Stub(
    "parallel_clip",
    image=modal.Image.debian_slim().pip_install("argparse", "moviepy", "numpy", "matplotlib", "tqdm", "joblib",
                                                "psutil", "duckdb"),
)


def add_output_file(output_file: str):
    return os.path.join("output", output_file)


class CustomVideoEditor(BaseModel):
    input_file: str
    output_file: str
    modal: bool
    only_plot: bool
    custom_scene: tuple[str, str, float]
    callback: Optional[Callable] = None

    def verify(self):
        if self.output_file is not None and self.only_plot:
            if self.callback:
                self.callback(CustomWebsocketData(type="PRINT",
                                                  message="--out will be ignored when --only_plot is set, remove output file"))

        if self.output_file and os.path.splitext(self.input_file)[1] != os.path.splitext(self.output_file)[1]:
            raise ValueError("Input and output files must have the same file extension.")

        if self.output_file is None and not self.only_plot:
            if self.callback:
                self.callback(CustomWebsocketData(
                    type="PRINT",
                    message="--out must specify an output file if --only_plot is not set! Will continue with tmp.mkv."))
            self.output_file = "tmp.mkv"

        if self.output_file and os.path.splitext(self.output_file)[1] != '.mkv':
            raise ValueError("Output file must have .mkv extension.")

    def run(self):
        self.output_file = add_output_file(self.output_file)
        if self.modal:
            with stub.run():
                self.process_input()
        else:
            self.process_input()

    @stub.local_entrypoint()
    def process_input(self):
        if self.only_plot:
            self.get_dimmed_scenes(0)
            return None

        if self.custom_scene:
            start, end, factor = self.custom_scene
            start, end = time_to_frame(start), time_to_frame(end)
            factor = float(factor) if float(factor) > 0.0 else self.get_dim_factor(start, end)
            dimmed_scenes = [(start, end, factor)]
        else:
            dimmed_scenes = self.get_all_dimmed_scenes()

        dimmed_scenes_timestamps = [(frame_to_time(start), frame_to_time(end), "{:.2f}".format(factor)) for
                                    start, end, factor in dimmed_scenes]

        if self.callback:
            self.callback(CustomWebsocketData(type="PRINT",
                                              message=f"Dimmed scenes (start time, stop time, dim factor): {dimmed_scenes_timestamps}"))

        plot_filename = self.output_file.replace('.mkv', '_dimmed_scenes_plot.png')
        self.plot_dimmed_scenes(dimmed_scenes_timestamps, plot_filename)
        self.process_video(dimmed_scenes)
        self.copy_subtitles()

    def get_dimmed_scenes(self, threshold: int) -> list:
        """
        Return (or plot) scene ranges with dim factors throughout the video.
        This function takes in an input video file and a boolean value for whether to show the plot or not.
        If show_plot, it only calculates the maximum frame value for each half second throughout the video and plots it.
        If show_plot is True, it will display the plot and return an empty list.
        If show_plot is False, it will not display the plot and will return a list of time ranges where at least 15 consecutive frames have a max below 190.
        The returned time ranges represent the dimmed scenes in the video.

        Parameters:
        input_file (str): The path to the input video file.
        show_plot (bool): Whether to display the plot or not.

        Returns:
        list: A list of time ranges (start frame, end frame, factor) representing the dimmed scenes in the video and how much to undim them by, if show_plot is False. An empty list if show_plot is True.
        """
        clip = VideoFileClip(self.input_file)
        max_values, max_no_outliers_values, avg_values, diff_values = self.load_values(clip)
        n_frames = 1
        max_values_per_n_frames = calculate_fn_per_frame_group(max_values, np.max, n_frames)
        max_no_outliers_values_values_per_n_frames = calculate_fn_per_frame_group(max_no_outliers_values, np.max,
                                                                                  n_frames)
        avg_values_per_n_frames = calculate_fn_per_frame_group(avg_values, np.mean, n_frames)

        if self.only_plot:
            self.plot_values(max_values_per_n_frames, avg_values_per_n_frames,
                             max_no_outliers_values_values_per_n_frames,
                             n_frames)
            return []

        dark_and_dimmed_ranges = self.find_dark_and_dimmed_ranges(max_values, threshold)
        dimmed_ranges = self.filter_and_print_range_characteristics(clip, max_values, avg_values, diff_values,
                                                                    dark_and_dimmed_ranges)
        return dimmed_ranges

    def get_dim_factor(self, start: int, end: int) -> float:
        """
        Given a time range, return the dim factor.

        Parameters:
        input_file (str): The path to the input video file.
        start (int): The start frame number of the time range.
        end (int): The end frame number of the time range.

        Returns:
        float: The dim factor for the given time range.
        """

        clip = VideoFileClip(self.input_file)
        max_values, max_no_outliers_values, avg_values, diff_values = self.load_values(clip)
        range_values = max_values[int(start):int(end)]
        avg_value = np.mean([np.max(val) for val in range_values])
        dim_factor = 256 / avg_value

        self.filter_and_print_range_characteristics(clip, max_values, avg_values, diff_values, [(int(start), int(end))])
        if self.callback:
            self.callback(CustomWebsocketData(type="PRINT", message=f"Dim factor auto calculated: {dim_factor}"))
        return dim_factor

    def get_all_dimmed_scenes(self):
        """
        Return (or plot) scene ranges with dim factors throughout the video for a variety of dim thresholds.
        This function takes in an input video file and a boolean value for whether to show the plot or not.
        If show_plot, it only calculates the maximum frame value for each half second throughout the video and plots it.
        If show_plot is True, it will display the plot and return an empty list.
        If show_plot is False, it will not display the plot and will return a list of time ranges where at least 15 consecutive frames have a max below 190.
        The returned time ranges represent the dimmed scenes in the video, ordered by most dim first.

        Parameters:
        input_file (str): The path to the input video file.
        show_plot (bool): Whether to display the plot or not.

        Returns:
        list: A list of time ranges (start frame, end frame, factor) representing the dimmed scenes in the video and how much to undim them by, if show_plot is False. An empty list if show_plot is True.
        """
        thresholds = [150, 190, 230]
        dimmed_scenes = []
        thresholds.sort()
        for threshold in thresholds:
            if self.callback:
                self.callback(CustomWebsocketData(type="PRINT",
                                                  message=f"---------CALCULATING FOR DIM PERCENT >= {((256 - threshold) / 256):.2f}----------------"))
            dimmed_scenes.extend(self.get_dimmed_scenes(threshold))
        return dimmed_scenes

    def plot_dimmed_scenes(self, dimmed_scenes_timestamps, filename):
        """
        Plot the dimmed scenes as a bar graph to visualize which scenes were dimmed the most.

        Parameters:
        dimmed_scenes_timestamps (list of tuples): List containing tuples of (start time, stop time, dim factor)
        """
        import matplotlib.pyplot as plt

        start_times = [start for start, _, _ in dimmed_scenes_timestamps]
        end_times = [end for _, end, _ in dimmed_scenes_timestamps]
        dim_factors = [float(factor) for _, _, factor in dimmed_scenes_timestamps]

        start_times_seconds = [int(min_sec.split(':')[0]) * 60 + float(min_sec.split(':')[1]) for min_sec in
                               start_times]
        end_times_seconds = [int(min_sec.split(':')[0]) * 60 + float(min_sec.split(':')[1]) for min_sec in end_times]

        durations = [end - start for start, end in zip(start_times_seconds, end_times_seconds)]

        plt.figure(figsize=(10, 6))

        for start, duration, factor in zip(start_times_seconds, durations, dim_factors):
            plt.bar(x=start, height=factor, width=duration, align='edge', alpha=0.7)

        original_ticks = plt.xticks()[0]
        new_ticks = np.arange(min(original_ticks), max(original_ticks) + 1,
                              (max(original_ticks) - min(original_ticks)) / (len(original_ticks) * 3 - 1))
        new_labels = [f"{int(s // 60)}:{int(s % 60):02d}" for s in new_ticks]
        plt.xticks(ticks=new_ticks, labels=new_labels, rotation=45, ha="right")
        plt.xlabel('Time (minutes:seconds)')
        plt.ylabel('Dim Factor')
        plt.title('Dimmed Scenes Visualization')
        plt.savefig(filename)
        plt.close()

    def process_video(self, dimmed_scenes):
        """
        Process the video, multiplying each frame's pixel values by the specified factor.
        Only multiply colors in the dimmed scenes range.
        """
        clip: VideoFileClip = VideoFileClip(self.input_file)
        clip = clip.fl(lambda gf, t: multiply_colors(gf(t), dimmed_scenes, int(t * clip.fps)))
        clip.write_videofile(self.output_file, codec='libx264', audio_codec='aac', threads=4)

    def copy_subtitles(self):
        """
        Adds subtitle tracks from an input MKV file to another MKV video file without altering the original video and audio tracks.
        This function creates a new output file that combines the original video and audio streams with the subtitle streams from the input subtitle file.

        Parameters:
        input_video_file (str): Path to the original video file whose video and audio streams will remain unchanged.
        subtitle_file (str): Path to the MKV file from which subtitles will be copied.
        output_file (str): Path to the new output file that will contain the combined streams.
        """
        subtitle_file = self.output_file.rsplit('.', 1)[0] + '_subtitled.' + self.output_file.rsplit('.', 1)[1]
        try:

            command = [
                'ffmpeg',
                '-y',
                '-i', self.output_file,
                '-i', self.input_file,
                '-map', '0:v',
                '-map', '0:a',
                '-map', '1:s',
                '-c', 'copy',
                '-c:s', 'copy',
                subtitle_file
            ]

            subprocess.run(command, check=True)

            shutil.move(subtitle_file, self.output_file)
            if self.callback:
                self.callback(CustomWebsocketData(type="PRINT",
                                                  message="Subtitles added successfully, video and audio preserved."))
        except subprocess.CalledProcessError as e:
            if self.callback:
                self.callback(CustomWebsocketData(type="PRINT", message=f"Failed to add subtitles: {e}"))

    def plot_values(self, max_values_per_n_frames, avg_values_per_n_frames, max_no_outliers_values_per_n_frames,
                    group_size=6):
        """
        Plot the max, max (no outliers), and average values (over each 6 frames, set by group_size) and show the plot.
        """
        plt.plot([x * group_size / 24 for x in range(len(max_values_per_n_frames))],
                 [np.max(val) for val in max_values_per_n_frames], label='Max')
        plt.plot([x * group_size / 24 for x in range(len(max_no_outliers_values_per_n_frames))],
                 [np.max(val) for val in max_no_outliers_values_per_n_frames], label='Max (no outliers)')
        plt.plot([x * group_size / 24 for x in range(len(avg_values_per_n_frames))],
                 [np.mean(val) for val in avg_values_per_n_frames], label='Avg')
        plt.title('Max, max no outliers, and avg frame value per quarter second')
        plt.legend()

        plt.show()

    def filter_and_print_range_characteristics(self, clip: VideoFileClip, max_values, avg_values, diff_values,
                                               dark_and_dimmed_ranges):
        """
        Print characteristics of the values in each range.
        """
        dimmed_ranges = []
        for start, end in dark_and_dimmed_ranges:
            range_max_values = max_values[int(start):int(end)]
            range_avg_values = avg_values[int(start):int(end)]
            range_diff_values = diff_values[int(start + 1):int(end)]
            result = self.get_and_print_single_range_characteristics(clip, start, end, range_max_values,
                                                                     range_avg_values,
                                                                     range_diff_values)
            if result:
                dimmed_ranges.append(result)
            if self.callback:
                self.callback(CustomWebsocketData(type="PRINT", message=""))

        return dimmed_ranges

    def find_dark_and_dimmed_ranges(self, max_values, threshold):
        """
        Find all the time ranges where at least 15 consecutive frames have a max below threshold.
        Iterative and not numpy optimized so is pretty slow.
        """
        start_timer = time.time()
        dark_and_dimmed_ranges = []
        count = 0
        start_time = 0
        for i, value in enumerate(max_values):
            if np.max(value) < threshold:
                if count == 0:
                    start_time = i
                count += 1
            else:
                if count >= 15:
                    dark_and_dimmed_ranges.append((start_time, i))
                count = 0
        if count >= 15:
            dark_and_dimmed_ranges.append(
                (start_time, len(max_values)))
        end_timer = time.time()
        if self.callback:
            self.callback(CustomWebsocketData(type="PRINT",
                                              message=f"Time taken by find_dark_and_dimmed_ranges (iterative): {(end_timer - start_timer) * 1000} milliseconds"))
        return dark_and_dimmed_ranges

    def load_values(self, clip):
        """
        Load the max, avg values and diff between each pair of consecutive scenes from the cache file if it exists, otherwise calculate them and store them in the cache file.
        """
        if self.callback:
            self.callback(CustomWebsocketData(type="PRINT", message="Starting analysis..."))
        max_values = []
        avg_values = []
        diff_values = []

        cache_file = f"{self.input_file}_max_avg_and_diff_values.pkl"

        if os.path.exists(cache_file):
            if self.callback:
                self.callback(CustomWebsocketData(type="PRINT", message="Cached!"))

            with open(cache_file, 'rb') as f:
                processed_values = pickle.load(f)
                if len(processed_values) == 3:
                    max_values, avg_values, diff_values = processed_values
                    max_no_outliers_values = max_values
                else:
                    max_values, max_no_outliers_values, avg_values, diff_values = processed_values

        else:
            if self.callback:
                self.callback(CustomWebsocketData(type="PRINT", message="Not cached! Calculating params now..."))
            max_values, max_no_outliers_values, avg_values, diff_values = self.process_clip_parallel(clip)
            if self.callback:
                self.callback(CustomWebsocketData(type="PRINT", message="Done calculating! Caching..."))
            with open(cache_file, 'wb') as f:
                pickle.dump((max_values, max_no_outliers_values, avg_values, diff_values), f)
            if self.callback:
                self.callback(CustomWebsocketData(type="PRINT", message=f"Cached! Delete {cache_file} to clear it."))

        if self.callback:
            self.callback(CustomWebsocketData(type="PRINT", message="Loaded max, avg and diff values!"))
        return max_values, max_no_outliers_values, avg_values, diff_values

    def get_and_print_single_range_characteristics(self, clip: VideoFileClip, start, end, range_max_values,
                                                   range_avg_values,
                                                   range_diff_values):
        if self.callback:
            self.callback(CustomWebsocketData(type="PRINT",
                                              message=f"Possible dark or dimmed time range: {frame_to_time(start)} - {frame_to_time(end)} minutes"))
        avg_value = np.mean([np.max(val) for val in range_max_values])
        max_value = np.max([np.max(val) for val in range_max_values])
        mean_value_no_outliers = calculate_mean_without_outliers(range_max_values)
        min_value = np.min([np.min(val) for val in range_max_values])
        variance = np.var([np.var(val) for val in range_max_values])
        if self.callback:
            self.callback(CustomWebsocketData(type="PRINT", message=
            f"Average value: {avg_value:.2f}, Max value: {max_value}, Min value: {min_value}, Mean without Outliers: {mean_value_no_outliers}, Variance: {variance:.2f}"))
        if len(range_max_values) > 0:
            filtered_values = [x for x in range_max_values if isinstance(x, np.ndarray) and x.shape == (3,)]
            if len(filtered_values) > 0:
                if self.callback:
                    self.callback(CustomWebsocketData(type="PRINT",
                                                      message=f"Variance between channels: {[round(var, 2) for var in np.var(filtered_values, axis=0)]}"))

        exact_frame_values = frame_generator(clip, start, end)
        exact_frame_values_2 = frame_generator(clip, start, end)
        is_epileptic = self.calculate_epilepsy_risk(exact_frame_values, exact_frame_values_2, range_max_values,
                                                    range_avg_values,
                                                    range_diff_values, 256 / avg_value)
        if is_epileptic:
            if self.callback:
                self.callback(CustomWebsocketData(type="PRINT",
                                                  message=f"Likely dimmed scene! Undimming range:  ({start}, {end}, {256 / avg_value:.2f})"))

            return (start, end, 256 / avg_value)
        else:
            if self.callback:
                self.callback(CustomWebsocketData(type="PRINT",
                                                  message=f"Likely NOT dimmed scene, just dark! If it was, dim range:  ({start}, {end}, {256 / avg_value:.2f})"))
            return None

    def process_clip_parallel(self, clip: VideoFileClip):
        """
        Process a clip in parallel and calculate the maximum, average and difference values for each frame.

        Parameters:
        clip (VideoFileClip): The video clip to be processed. It's an object of class VideoFileClip.

        Returns:
        tuple: A tuple containing lists of maximum (1D array of RGB values), maximum without outliers (1D array of RGB values), average (1D array of RGB values) and difference values (1D array of single float values) for each frame in the clip.
        """
        cores = psutil.cpu_count(logical=False)
        batch_size = 8
        if self.callback:
            self.callback(
                CustomWebsocketData(type="PRINT", message=f"Using {batch_size} batch size, {cores} cores available..."))

        with Parallel(n_jobs=batch_size) as parallel:
            results = []
            frame_generator = clip.iter_frames(with_times=True)
            prev_frame = None

            total_frames = int(clip.fps * clip.duration)

            # pbar = tqdm(total=total_frames, desc="Processing frames")
            updates = 0
            if self.callback:
                self.callback(CustomWebsocketData(type="PROGRESS", message=0, desc="Processing frames"))

            while True:

                batch = list(itertools.islice(frame_generator, batch_size))
                if not batch:
                    break

                batch_jobs = [(frame, prev_frame if i > 0 else frame) for i, (t, frame) in enumerate(batch)]

                batch_results = Parallel(n_jobs=batch_size)(
                    delayed(self.process_frame_parallel)(*job) for job in batch_jobs)

                results.extend(batch_results)

                prev_frame = batch[-1][1]
                updates += len(batch)
                if self.callback:
                    self.callback(CustomWebsocketData(type="PROGRESS", message=updates / total_frames))
                # pbar.update(len(batch))

        # pbar.close()
        max_values, max_no_outliers_values, avg_values, diff_values = zip(*results)
        return max_values, max_no_outliers_values, avg_values, diff_values

    def calculate_epilepsy_risk(self, frame_values_gen, frame_values_gen_2, range_max_values, range_avg_values,
                                range_diff_values,
                                dim_multiplier):
        """
        Calculate the risk of epilepsy for a video.

        This function calculates the mean and standard deviation of the absolute sum of differences between consecutive frames.
        A high mean and standard deviation indicates a high risk of epilepsy.

        Parameters:
        frame_values_gen (generator): A generator that yields the pixel values for each frame in the video.

        Returns:
        mean, stddev: A tuple containing the mean and standard deviation of the absolute sum of differences between consecutive frames.
        """

        abs_sum_diffs = range_diff_values
        abs_luminescance = np.mean(range_avg_values, axis=1)

        frame_count = len(abs_sum_diffs)

        abs_sum_diffs = np.array(abs_sum_diffs)

        flash_count = np.sum(abs_sum_diffs > 20)

        flash_count_corrected = np.sum(abs_sum_diffs > (20 / dim_multiplier))

        is_flash = abs_sum_diffs > (20 / dim_multiplier)

        flash_indices = np.where(is_flash)[0]

        flash_diffs = np.diff(flash_indices)

        close_flash_count = np.sum(flash_diffs < 9)

        flash_count_below_160 = np.sum(
            (abs_sum_diffs > 20) & ((np.array(abs_luminescance[:-1]) < 160) | (np.array(abs_luminescance[1:]) < 160)))

        dimmed_160 = 160 / dim_multiplier
        flash_count_below_160_corrected = np.sum((abs_sum_diffs > (20 / dim_multiplier)) & (
                (np.array(abs_luminescance[:-1]) < dimmed_160) | (np.array(abs_luminescance[1:]) < dimmed_160)))

        risk_mean = np.mean(abs_sum_diffs)

        risk_stddev = np.std(abs_sum_diffs)

        if self.callback:
            self.callback(CustomWebsocketData(type="PRINT", message=f"Epileptic risk: {risk_mean:.1f}, {risk_stddev:.1f}. \
    Flashes: {flash_count / frame_count:.2f}, {flash_count} in {frame_count} frames, \
    Flashes with a <160: {flash_count_below_160 / frame_count:.2f}, {flash_count_below_160} in {frame_count} frames, \
    Predim flashes: {flash_count_corrected / frame_count:.2f}, {flash_count_corrected} in {frame_count} frames, \
    Predim flashes with a predim <160: {flash_count_below_160_corrected / frame_count:.2f}, {flash_count_below_160_corrected} in {frame_count} frames, \
    Predim flashes less than 9 frames apart: {close_flash_count / frame_count:.2f}, {close_flash_count} in {frame_count} frames"))

        if risk_mean > 10 or flash_count >= 3:
            return True
        return False


def calculate_mean_without_outliers(values) -> float:
    """
    This function calculates the mean of the luminescence of each frame, removing outliers.

    Parameters:
    values (np.array): A numpy array of luminescence values of each frame. Each frame can be a full RGB frame or a single value.

    Returns:
    float: The mean of the luminescence values after removing outliers.
    """
    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25
    threshold_values = [x for x in values if ((q25 - 1.5 * iqr) <= np.max(x) <= (q75 + 1.5 * iqr))]
    return np.mean([np.max(val) for val in threshold_values])


def time_to_frame(time_str: str) -> int:
    if ':' in time_str:
        minutes, seconds = map(float, time_str.split(':'))
    else:
        minutes = 0
        seconds = float(time_str)
    return int((minutes * 60 + seconds) * 24)


def frame_to_time(frame_num: int) -> str:
    """
    Convert frame number to timestamp in minute:second format.
    """
    minutes = frame_num // (24 * 60)
    seconds = (frame_num / 24) % 60
    return f"{minutes:02d}:{seconds:.2f}"


def multiply_colors(frame, dimmed_scenes, current_frame):
    """
    Multiply each color in the frame by a given factor.
    Clipping is performed to ensure pixel values stay within valid range.
    Only multiply colors in the dimmed scenes range.
    """
    for start, end, factor in dimmed_scenes:
        if start <= current_frame < end:
            return np.clip(frame * factor, 0, 255).astype('uint8')
    return frame


def calculate_fn_per_frame_group(max_values, fn=np.max, frames=6):
    """
    Calculate fn on every 6 frames in max_values.
    """
    fn_over_frame_group = []
    for i in range(0, len(max_values), frames):
        frames_to_average = max_values[i:i + 6]
        max_frame = fn(frames_to_average, axis=0)
        fn_over_frame_group.append(max_frame)
    return fn_over_frame_group


def frame_generator(clip, start, end):
    start_frame = int(start)
    end_frame = int(end)
    for i, frame in enumerate(clip.iter_frames()):
        if i < start_frame:
            continue
        if i >= end_frame:
            break
        yield frame
