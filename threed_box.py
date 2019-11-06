import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import Polygon
import math

class Box3D:
    """Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self, x, y, z, xl, yl, zl, r):
        

        

        # Assign.
        

        self.volume = xl * yl * zl
        self.r = r
        self.quaternion = Quaternion(r)

        self.width, self.length, self.height = xl, yl, zl

        self.center_x, self.center_y, self.center_z = x, y, z

        self.min_z = self.center_z - self.height / 2
        self.max_z = self.center_z + self.height / 2

        self.ground_bbox_coords = None
        self.ground_bbox_coords = self.get_ground_bbox_coords()

    @staticmethod
    def check_orthogonal(a, b, c):
        """Check that vector (b - a) is orthogonal to the vector (c - a)."""
        return np.isclose((b[0] - a[0]) * (c[0] - a[0]) + (b[1] - a[1]) * (c[1] - a[1]), 0)

    def get_ground_bbox_coords(self):
        if self.ground_bbox_coords is not None:
            return self.ground_bbox_coords
        return self.calculate_ground_bbox_coords()

    def calculate_ground_bbox_coords(self):
        """We assume that the 3D box has lower plane parallel to the ground.

        Returns: Polygon with 4 points describing the base.

        """
        if self.ground_bbox_coords is not None:
            return self.ground_bbox_coords

        #rotation_matrix = self.quaternion.rotation_matrix
        #print(rotation_matrix)
        cos_angle = math.cos(math.radians(self.r))
        sin_angle = math.sin(math.radians(self.r))
        point_0_x = self.center_x + self.length / 2 * cos_angle + self.width / 2 * sin_angle
        point_0_y = self.center_y + self.length / 2 * sin_angle - self.width / 2 * cos_angle

        point_1_x = self.center_x + self.length / 2 * cos_angle - self.width / 2 * sin_angle
        point_1_y = self.center_y + self.length / 2 * sin_angle + self.width / 2 * cos_angle

        point_2_x = self.center_x - self.length / 2 * cos_angle - self.width / 2 * sin_angle
        point_2_y = self.center_y - self.length / 2 * sin_angle + self.width / 2 * cos_angle

        point_3_x = self.center_x - self.length / 2 * cos_angle + self.width / 2 * sin_angle
        point_3_y = self.center_y - self.length / 2 * sin_angle - self.width / 2 * cos_angle

        point_0 = point_0_x, point_0_y
        point_1 = point_1_x, point_1_y
        point_2 = point_2_x, point_2_y
        point_3 = point_3_x, point_3_y
#        print(point_0)

#         assert self.check_orthogonal(point_0, point_1, point_3)
#         assert self.check_orthogonal(point_1, point_0, point_2)
#         assert self.check_orthogonal(point_2, point_1, point_3)
#         assert self.check_orthogonal(point_3, point_0, point_2)
#         print(point_0)
#         print(point_1)
#         print(point_2)
#         print(point_3)
        self.ground_bbox_coords = Polygon(
            [
                (point_0_x, point_0_y),
                (point_1_x, point_1_y),
                (point_2_x, point_2_y),
                (point_3_x, point_3_y),
                (point_0_x, point_0_y),
            ]
        )

        return self.ground_bbox_coords

    def get_height_intersection(self, other):
        min_z = max(other.min_z, self.min_z)
        max_z = min(other.max_z, self.max_z)

        return max(0, max_z - min_z)

    def get_area_intersection(self, other) -> float:
        result = self.ground_bbox_coords.intersection(other.ground_bbox_coords).area

        assert result <= self.width * self.length

        return result

    def get_intersection(self, other) -> float:
        height_intersection = self.get_height_intersection(other)

        area_intersection = self.ground_bbox_coords.intersection(other.ground_bbox_coords).area

        return height_intersection * area_intersection

    def get_iou(self, other):
        intersection = self.get_intersection(other)
        union = self.volume + other.volume - intersection
        #print(intersection / union)
        iou = np.clip(intersection / union, 0, 1)

        return iou

    def __repr__(self):
        return str(self.serialize())

    def serialize(self) -> dict:
        """Returns: Serialized instance as dict."""

        return {
            "sample_token": self.sample_token,
            "translation": self.translation,
            "size": self.size,
            "rotation": self.rotation,
            "name": self.name,
            "volume": self.volume,
            "score": self.score,
        }


def group_by_key(detections, key):
    groups = defaultdict(list)
    for detection in detections:
        groups[detection[key]].append(detection)
    return groups


def wrap_in_box(input):
    result = {}
    for key, value in input.items():
        result[key] = [Box3D(**x) for x in value]

    return result



def evatwobox(predict, ground, thd):
    boxa = Box3D(float(predict[1]), float(predict[2]), float(predict[3]), float(predict[4]),
                float(predict[5]), float(predict[6]), float(predict[7]))
    boxb = Box3D(float(ground[0]), float(ground[1]), float(ground[2]), float(ground[3]),
                float(ground[4]), float(ground[5]), float(ground[6]))
    if boxa.get_iou(boxb) > thd:
        return True



