from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type, TypeVar

from rtnls_enface.base import Layer
from vascx.shared.segment import Segment

T = TypeVar("T")


def get_all_subclasses_dict(cls: Type[T]) -> Dict[str, Type[T]]:
    all_subclasses_dict = {}

    for subclass in cls.__subclasses__():
        all_subclasses_dict[subclass.__name__] = subclass
        all_subclasses_dict.update(get_all_subclasses_dict(subclass))

    return all_subclasses_dict


class VesselLayer(Layer, ABC):
    @property
    @abstractmethod
    def binary(self):
        pass

    @property
    @abstractmethod
    def skeleton(self):
        pass

    @property
    @abstractmethod
    def graph(self):
        pass

    @property
    @abstractmethod
    def trees(self):
        pass

    @property
    @abstractmethod
    def digraph(self):
        pass

    @property
    @abstractmethod
    def segments(self):
        pass

    @property
    @abstractmethod
    def nodes(self):
        pass

    @abstractmethod
    def get_segment_pixels(self, segment: Segment) -> List[Tuple[int, int]]:
        """Returns the binary mask pixels associated to the given segment."""
        pass


class JointVesselsLayer(Layer, ABC):
    @property
    @abstractmethod
    def binary(self):
        pass

    @property
    @abstractmethod
    def skeleton(self):
        pass
