from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass
