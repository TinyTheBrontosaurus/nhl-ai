import neat
import time
from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys

import gzip
import random

try:
    import cPickle as pickle # pylint: disable=import-error
except ImportError:
    import pickle # pylint: disable=import-error

class TqdmReporter(neat.reporting.BaseReporter):

    def __init__(self, progress_bar, stream=None):
        """
        Reporter that prints out TQDM for each generation
        :param progress_bar: The TQDM progress bar
        """
        self.progress_bar = progress_bar
        self.stream = stream
        self._last_fitness = None
        self._fitness_stall = 0
        self._num_extinctions = 0
        self._uniques = 0

    def end_generation(self, config, population, species_set):
        self.update()

    def found_solution(self, config, generation, best):
        self.update()

    def post_evaluate(self, config, population, species, best_genome):
        if self._last_fitness != best_genome.fitness:
            self._last_fitness = best_genome.fitness
            self._fitness_stall = 0
            self._uniques += 1
        else:
            self._fitness_stall += 1

        self.progress_bar.set_postfix(fitness="{:.0f}/{:.0f}".format(best_genome.fitness, config.fitness_threshold),
                                      progress="{:.2f}%".format(best_genome.fitness / config.fitness_threshold * 100),
                                      stall=self._fitness_stall,
                                      uniques=self._uniques,
                                      extinctions=int(self._num_extinctions))

    def complete_extinction(self):
        self._num_extinctions += 1
        self.stream('All species extinct.')

    def update(self):
        if self.progress_bar is not None:
            self.progress_bar.update()
            if self.stream is not None:
                self.stream(str(self.progress_bar))

    def __getstate__(self):
        """
        Prevent tqdm failing on a pickle during checkpoint
        """
        return None

    def __setstate__(self, _):
        """
        There will be no progress bar when returning from a picle
        """
        self.progress_bar = None


class GenerationReporter(neat.reporting.BaseReporter):
    """See StdOutReporter
    Same as StdOutReporter except it takes in a custom stream for printing"""
    def __init__(self, show_species_detail, stream=print):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.stream = stream

    def start_generation(self, generation):
        self.generation = generation
        self.stream('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            self.stream('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            sids = list(iterkeys(species_set.species))
            sids.sort()
            self.stream("   ID   age  size  fitness  adj fit  stag")
            self.stream("  ====  ===  ====  =======  =======  ====")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                self.stream(
                    "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st))
        else:
            self.stream('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        self.stream('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            self.stream("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            self.stream("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        self.stream('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        self.stream(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

    def complete_extinction(self):
        self.num_extinctions += 1
        self.stream('All species extinct.')

    def found_solution(self, config, generation, best):
        self.stream('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            self.stream("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg):
        self.stream(msg)


class SaveBestOfGeneration(neat.reporting.BaseReporter):
    def __init__(self, prefix: str):
        """
        Save the best genome at the end of each generation
        :param prefix: The prefix for the genomes being saved. Each will have "{id}.pkl"
        added as a suffix
        """
        self.generation: int = None
        self.prefix: str = prefix
        self.last_logged = None

    def start_generation(self, generation: int):
        """
        Capture the ID of the next generation
        """
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):

        if best_genome is not None:
            # Only log new genomes; don't log the same genome over and over again when stalled
            new_best_species_id = species.get_species_id(best_genome.key)
            new_best_genome_key = best_genome.key

            if self.last_logged is None:
                log_it = True
            elif new_best_species_id == self.last_logged[0] and new_best_genome_key == self.last_logged[1]:
                log_it = False
            else:
                log_it = True

            if log_it:
                self.last_logged = (new_best_species_id, new_best_genome_key)

                # Dump the result
                model_filename = "{}{}.pkl".format(self.prefix, self.generation)
                with open(model_filename, 'wb') as f:
                    pickle.dump(best_genome, f, 1)


class Checkpointer(neat.checkpoint.Checkpointer):
    def __init__(self, generation_interval=100, time_interval_seconds=300,
                 filename_prefix='neat-checkpoint-', stream=print):
        super().__init__(generation_interval, time_interval_seconds, filename_prefix)
        self.stream = stream

    def save_checkpoint(self, config, population, species_set, generation):
        """ Save the current simulation state. """
        filename = '{0}{1}'.format(self.filename_prefix,generation)
        self.stream("Saving checkpoint to {0}".format(filename))

        with gzip.open(filename, 'w', compresslevel=5) as ff:
            data = (generation, config, population, species_set, random.getstate())
            pickle.dump(data, ff, protocol=pickle.HIGHEST_PROTOCOL)


