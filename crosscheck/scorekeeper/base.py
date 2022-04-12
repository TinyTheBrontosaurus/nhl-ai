import abc

class Scorekeeper:

    def __init__(self):
        self._done_reasons = {}
        self._ever_done = False
        self.info: dict = {}
        self._score = 0
        self._score_vector = {}
        self._stats = {}
        self.buttons_pressed: dict = {}

        # For compatibility with genome stats puller
        self._scorekeepers = []

    def tick(self) -> float:
        self._score = self._tick()
        return self._score

    @property
    def score(self) -> float:
        """
        Accessor for the score calculated in tick
        """
        return self._score

    @property
    def done(self) -> bool:
        """
        :return: True if the scorekeeper considers the scenario complete
        """
        # Check if done this time
        is_done = any(self._done_reasons.values())
        # Latch the done signal to yes
        self._ever_done = is_done or self._ever_done
        return self._ever_done

    def done_reasons(self) -> dict:
        """
        :return: The reasons, as keys in a dictionary, indicating
        the triggers that the scorekeeper monitors to declare a
        scenario complete
        """
        return self._done_reasons

    @property
    def score_vector(self) -> dict:
        return self._score_vector

    @property
    def stats(self) -> dict:
        return self._stats

    @abc.abstractmethod
    def _tick(self) -> float:
        """
        Incorporate the latest frame into the total score
        :return: The total score as of this frame
        """
        pass

    @classmethod
    @abc.abstractclassmethod
    def fitness_threshold(cls) -> float:
        """
        Accessor fitness threshold (the score over
        which to stop training)
        """
        pass

