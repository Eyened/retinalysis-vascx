from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from inspect import signature
from typing import List, Set, Union

from vascx.layer import Layer
from vascx.segment import Segment


class Bifurcation:

    def __init__(self, incoming: List[Segment], outgoing: List[Segment]):
        self.incoming = incoming
        self.outgoing = outgoing
