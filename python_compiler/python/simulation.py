from oncosim import *
from enum import Enum


class State(Enum):
    Necrotic = 0
    Queicent = 1
    Proliferating = 2


class CellData:
    idx: int
    type_id: int
    state: int
    energy: float


class EnvironmentData:
    type_id: int


class FieldData:
    o2: float
    glu: float


class EnvironmentPoint(EnvironmentData):
    @environment_rule
    def update(self, cells, fields):
        pass


class FieldPoint(FieldData):
    @reaction_rule
    def update(self, cells, env):
        pass

    @diffusion_rule(2)
    def diffusion(self, neighbors: list["FieldPoint"]):
        pass


class Cell:
    @update_rule
    def update(self, neighbors, env, fields):
        if self.energy < 5.0:
            self.state = State.Necrotic    

    @division_rule
    def spawn(self) -> "Cell" | None:
        if self.state == State.Proliferating and self.energy > 90.0:
            self.energy *= 0.5
            return self.clone()
        return None
    
    def clone(self):
        return Cell(
            idx=self.idx,
            type=self.type,
            state=self.state,
            energy=self.energy)


run()
