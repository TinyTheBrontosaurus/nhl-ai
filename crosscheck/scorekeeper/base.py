import abc

class Scorekeeper:

    def __init__(self):
        self._done_reasons = {}
        self.info = None

    @abc.abstractmethod
    def tick(self):
        """
        Incorporate the latest frame into the total score
        :return: The total score as of this frame
        """
        pass

    @property
    def done(self):
        """
        :return: True if the scorekeeper considers the scenario complete
        """
        return any(self._done_reasons.items())

    def done_reasons(self) -> dict:
        """
        :return: The reasons, as keys in a dictionary, indicating
        the triggers that the scorekeeper monitors to declare a
        scenario complete
        """
        return self._done_reasons

    @property
    def stats(self):
        pass

    @property
    def score_vector(self):
        pass