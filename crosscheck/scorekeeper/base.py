import abc

class Scorekeeper:

    def __init__(self):
        self._done_reasons = {}
        self.info = None
        self._score = 0

    def tick(self) -> float:
        self._score = self._tick()
        return self._score

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
        return any(self._done_reasons.values())

    def done_reasons(self) -> dict:
        """
        :return: The reasons, as keys in a dictionary, indicating
        the triggers that the scorekeeper monitors to declare a
        scenario complete
        """
        return self._done_reasons

    @property
    @abc.abstractmethod
    def stats(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def score_vector(self) -> dict:
        pass
