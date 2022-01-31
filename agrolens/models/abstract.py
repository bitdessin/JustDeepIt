from abc import ABCMeta
from abc import abstractmethod


class ModuleTemplate(metaclass = ABCMeta):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def inference(self):
        pass


