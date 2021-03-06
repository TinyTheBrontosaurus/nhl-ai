import confuse
import argparse
import pathlib
import sys
from typing import List, Callable, Type
from loguru import logger
from crosscheck import main_train
from crosscheck.config import cc_config
import imageio
import crosscheck.config
from crosscheck.neat_.replayer import Replayer
from crosscheck.version import __version__
from PIL import Image, ImageDraw
import numpy as np
import functools
import pickle
import natsort
import tqdm
from crosscheck import definitions
from crosscheck.log_folder import LogFolder

def main(argv):
    parser = argparse.ArgumentParser(description='Cross-check: NHL \'94 reinforcement learning')
    parser.add_argument('--folder', type=str, help="The folder to convert to a movie")

    args = parser.parse_args(argv)

    folder_to_load = args.folder

    if args.folder is None:
        log_folder = definitions.LOG_ROOT / crosscheck.config.log_name
        folder_to_load = LogFolder.get_latest_log_folder(log_folder)

    # Find the folder(s) to generate
    folders = [pathlib.Path(folder_to_load)]

    # Iterate over the folders
    for folder in folders:

        # Parse the config using confuse, bailing on failure
        try:

            # Load the config
            config_filename = pathlib.Path(folder) / "config.yml"

            cc_config.set_file(config_filename)

            _ = cc_config.get(main_train.template)

            replay(folder)

        except confuse.ConfigError as ex:
            logger.critical("Problem parsing config: {}", ex)
            return


def replay(folder: pathlib.Path):
    discretizer = main_train.load_discretizer(cc_config['input']['controller-discretizer'].get())
    feature_vector = main_train.load_feature_vector(cc_config['input']['feature-vector'].get())
    scenarios = main_train.load_scenarios(cc_config['input']['scenarios'])
    combiner = main_train.load_metascorekeeper(cc_config['input']['metascorekeeper'].get())
    for scenario in scenarios:

        # Create the movie
        movie_folder = folder / "movies"
        movie_folder.mkdir(exist_ok=True)
        with imageio.get_writer(str(movie_folder / f'training-{scenario.name}.mp4'), 'ffmpeg', fps=60) as movie:

            # Load the genomes
            generation_folder = folder / "generations"
            generation_files = natsort.natsorted([x
                                for x in generation_folder.iterdir()
                                if x.stem.startswith('generation-')])  # type: List[pathlib.Path]

            metadata = {
                "timestamp": None,
                "scenario": scenario,
            }

            replayer = Replayer(scenario, combiner, feature_vector,
                                str(folder / "neat_config.ini"), discretizer)
            replayer.listeners.append(functools.partial(add_frame, movie, metadata))

            with tqdm.tqdm(smoothing=0, unit='generation', total=len(generation_files)) as progress_bar:
                for generationi, genome_file in enumerate(generation_files):
                    metadata["generation"] = f"{generationi + 1}/{len(generation_files)}"
                    with genome_file.open(mode='rb') as f:
                        genome = pickle.load(f)
                    replayer.replay(genome)
                    progress_bar.update()


def add_frame(movie, metadata, ob, _rew, _done, _info, stats):
    blank_frame = np.zeros(ob.shape, dtype=np.uint8)
    img = Image.fromarray(blank_frame)
    draw = ImageDraw.Draw(img)

    sk = stats['scorekeeper']

    to_draw = dict(metadata)
    to_draw['version'] = __version__
    to_draw.update(sk.stats)

    score_vector = sk.score_vector
    total = sum(score_vector.values())
    score_breakdown = {"total": int(total)}

    for key, value in score_vector.items():
        try:
            pct = value / total * 100.
        except ZeroDivisionError:
            pct = 0
        score_breakdown[key] = "({pct:5,.1f}% {value:8,}".format(pct=pct, value=int(value))

    to_draw.update(score_breakdown)

    for offset, (key, value) in enumerate(to_draw.items()):
        draw.text((0, 5 + 12 * offset), "{:15}: {}".format(key, value), fill='rgb(255, 255, 255)')

    status_frame = np.array(img)

    new_ob = np.concatenate((ob, status_frame), axis=1)
    movie.append_data(new_ob)


if __name__ == "__main__":
    main(sys.argv[1:])