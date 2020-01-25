import confuse
import argparse
import pathlib
from typing import List, Callable, Type
from loguru import logger
from . import main_train
from .config import cc_config
import imageio
from .neat_.replayer import Replayer
from .version import __version__
from PIL import Image, ImageDraw
import numpy as np
import functools
import pickle
import natsort


def main(argv):
    parser = argparse.ArgumentParser(description='Cross-check: NHL \'94 reinforcement learning')
    parser.add_argument('--latest-only',
                        action='store_const',
                        const="latest",
                        dest="movie.enabled",
                        help="Only replay the latest training model")

    args = parser.parse_args(argv)

    # TODO: Find the folder(s) to generate
    folders = []

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
            generation_files = natsort.natsorted([generation_folder / x
                                for x in generation_folder.iterdir()
                                if str(x).startswith('generation')])

            metadata = {
                "timestamp": None,
                "scenario": scenario,
            }

            replayer = Replayer(scenario, combiner, feature_vector,
                                str(folder / "neat_config.ini"), discretizer)
            replayer.listeners.append(functools.partial(add_frame, movie, metadata))

            for generationi, genome_file in enumerate(generation_files):
                metadata["generation"] = f"{generationi}/{len(generation_files)}"
                with genome_file.open() as f:
                    genome = pickle.load(f)
                replayer.replay(genome)


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
        score_breakdown[key] = "({pct:5,.1f}% {value:8,}".format(pct=value / total * 100., value=int(value))

    to_draw.update(score_breakdown)

    for offset, (key, value) in enumerate(to_draw.items()):
        draw.text((0, 5 + 12 * offset), "{:15}: {}".format(key, value), fill='rgb(255, 255, 255)')

    status_frame = np.array(img)

    new_ob = np.concatenate((ob, status_frame), axis=1)
    movie.append_data(new_ob)
