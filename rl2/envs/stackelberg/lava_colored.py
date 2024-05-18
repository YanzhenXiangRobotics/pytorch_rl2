from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import (
    fill_coords,
    point_in_line,
    point_in_rect,
)

class LavaColored(WorldObj):
    def __init__(self, first_step):
        self.first_step = first_step
        if first_step:
            super().__init__("lava", "green")
        else:
            super().__init__("lava", "yellow")

    def can_overlap(self):
        return True

    def render(self, img):
        if self.first_step:
            c = (0, 128, 0)
        else:
            c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))