import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import os
import cv2
from pyquaternion import Quaternion
from shapely.geometry import Polygon
import math

classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]

def move_boxes_to_car_space(boxes, ego_pose):
    """
    Move boxes from world space to car space.
    Note: mutates input boxes.
    """
    translation = -np.array(ego_pose['translation'])
    rotation = Quaternion(ego_pose['rotation']).inverse

    for box in boxes:
        # Bring box to car space
        box.translate(translation)
        box.rotate(rotation)

def scale_boxes(boxes, factor):
    """
    Note: mutates input boxes
    """
    for box in boxes:
        box.wlh = box.wlh * factor

def draw_boxes(im, voxel_size, boxes, classes, z_offset=0.0):
    for box in boxes:
        # We only care about the bottom corners
        corners = box.bottom_corners()
        corners_voxel = car_to_voxel_coords(corners, im.shape, voxel_size, z_offset).transpose(1,0)
        corners_voxel = corners_voxel[:,:2] # Drop z coord

        class_color = classes.index(box.name) + 1

        if class_color == 0:
            raise Exception("Unknown class: {}".format(box.name))

        cv2.drawContours(im, np.int0([corners_voxel]), 0, (class_color, class_color, class_color), -1)


class BEVImageDataset(torch.utils.data.Dataset):
    def __init__(self, sample_tokens, data_folder, is_train=True):
        self.is_train = is_train
        self.sample_tokens = sample_tokens
        self.data_folder = data_folder

    def __len__(self):
        return len(self.sample_tokens)

    def __getitem__(self, idx):
        sample_token = self.sample_tokens[idx]

        # Get file path, assuming data already preprocessed and stored in data_folder
        input_filepath = os.path.join(self.data_folder,f"{sample_token}_input.png")
        map_filepath = os.path.join(self.data_folder,f"{sample_token}_map.png")

        target = None
        if self.is_train:
            target_filepath = os.path.join(self.data_folder,f"{sample_token}_target.png")
            target = cv2.imread(target_filepath, cv2.IMREAD_UNCHANGED)
            target = target.astype(np.int64)
            target = torch.from_numpy(target)

        im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)
        map_im = cv2.imread(map_filepath, cv2.IMREAD_UNCHANGED)
        # Concatenate image and map for network input
        im = np.concatenate((im, map_im), axis=2)

        im = im.astype(np.float32)/255
        im = torch.from_numpy(im.transpose(2,0,1))

        if not self.is_train:
            return im, sample_token
        else:
            return im, target, sample_token


def box_in_image(box, intrinsic, image_size) -> bool:
    """Check if a box is visible inside an image without accounting for occlusions.
    Args:
        box: The box to be checked.
        intrinsic: <float: 3, 3>. Intrinsic camera matrix.
        image_size: (width, height)
        vis_level: One of the enumerations of <BoxVisibility>.
    Returns: True if visibility condition is satisfied.
    """

    corners_3d = box.corners()
    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < image_size[0])
    visible = np.logical_and(visible, corners_img[1, :] < image_size[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    visible = np.logical_and(visible, corners_3d[2, :] > 1)

    in_front = corners_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    return any(visible) and all(in_front)


def viz_unet(sample_token,boxes):

    sample = level5data.get("sample", sample_token)

    sample_camera_token = sample["data"]["CAM_FRONT"]
    camera_data = level5data.get("sample_data", sample_camera_token)
    # camera_filepath = level5data.get_sample_data_path(sample_camera_token)

    ego_pose = level5data.get("ego_pose", camera_data["ego_pose_token"])
    calibrated_sensor = level5data.get("calibrated_sensor", camera_data["calibrated_sensor_token"])
    data_path, _, camera_intrinsic = level5data.get_sample_data(sample_camera_token)


    data = Image.open(data_path)
    _, axis = plt.subplots(1, 1, figsize=(9, 9))

    for i,box in enumerate(boxes):

        # Move box to ego vehicle coord system
        box.translate(-np.array(ego_pose["translation"]))
        box.rotate(Quaternion(ego_pose["rotation"]).inverse)

        # Move box to sensor coord system
        box.translate(-np.array(calibrated_sensor["translation"]))
        box.rotate(Quaternion(calibrated_sensor["rotation"]).inverse)

        if box_in_image(box,camera_intrinsic,np.array(data).shape):
            box.render(axis,camera_intrinsic,normalize=True)

    axis.imshow(data)
    all_pred_fn.append(f'./cam_viz/cam_preds_{sample_token}.jpg')
    plt.savefig(all_pred_fn[-1])
    plt.close()




def calc_detection_box(prediction_opened,class_probability):

    sample_boxes = []
    sample_detection_scores = []
    sample_detection_classes = []

    contours, hierarchy = cv2.findContours(prediction_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)

        # Let's take the center pixel value as the confidence value
        box_center_index = np.int0(np.mean(box, axis=0))

        for class_index in range(len(classes)):
            box_center_value = class_probability[class_index+1, box_center_index[1], box_center_index[0]]

            # Let's remove candidates with very low probability
            if box_center_value < 0.01:
                continue

            box_center_class = classes[class_index]

            box_detection_score = box_center_value
            sample_detection_classes.append(box_center_class)
            sample_detection_scores.append(box_detection_score)
            sample_boxes.append(box)

    return np.array(sample_boxes),sample_detection_scores,sample_detection_classes


def create_transformation_matrix_to_voxel_space(shape, voxel_size, offset):
    """
    Constructs a transformation matrix given an output voxel shape such that (0,0,0) ends up in the center.
    Voxel_size defines how large every voxel is in world coordinate, (1,1,1) would be the same as Minecraft voxels.

    An offset per axis in world coordinates (metric) can be provided, this is useful for Z (up-down) in lidar points.
    """

    shape, voxel_size, offset = np.array(shape), np.array(voxel_size), np.array(offset)

    tm = np.eye(4, dtype=np.float32)
    translation = shape/2 + offset/voxel_size

    tm = tm * np.array(np.hstack((1/voxel_size, [1])))
    tm[:3, 3] = np.transpose(translation)
    return tm

def transform_points(points, transf_matrix):
    """
    Transform (3,N) or (4,N) points using transformation matrix.
    """
    if points.shape[0] not in [3,4]:
        raise Exception("Points input should be (3,N) or (4,N) shape, received {}".format(points.shape))
    return transf_matrix.dot(np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]


def car_to_voxel_coords(points, shape, voxel_size, z_offset=0):
    if len(shape) != 3:
        raise Exception("Voxel volume shape should be 3 dimensions (x,y,z)")

    if len(points.shape) != 2 or points.shape[0] not in [3, 4]:
        raise Exception("Input points should be (3,N) or (4,N) in shape, found {}".format(points.shape))

    tm = create_transformation_matrix_to_voxel_space(shape, voxel_size, (0, 0, z_offset))
    p = transform_points(points, tm)
    return p

def create_voxel_pointcloud(points, shape, voxel_size=(0.5,0.5,1), z_offset=0):

    points_voxel_coords = car_to_voxel_coords(points.copy(), shape, voxel_size, z_offset)
    points_voxel_coords = points_voxel_coords[:3].transpose(1,0)
    points_voxel_coords = np.int0(points_voxel_coords)

    bev = np.zeros(shape, dtype=np.float32)
    bev_shape = np.array(shape)

    within_bounds = (np.all(points_voxel_coords >= 0, axis=1) * np.all(points_voxel_coords < bev_shape, axis=1))

    points_voxel_coords = points_voxel_coords[within_bounds]
    coord, count = np.unique(points_voxel_coords, axis=0, return_counts=True)

    # Note X and Y are flipped:
    bev[coord[:,1], coord[:,0], coord[:,2]] = count

    return bev

def normalize_voxel_intensities(bev, max_intensity=16):
    return (bev/max_intensity).clip(0,1)

class box3d:
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
    boxa = box3d(float(predict[1]), float(predict[2]), float(predict[3]), float(predict[4]),
                float(predict[5]), float(predict[6]), float(predict[7]))
    boxb = box3d(float(ground[0]), float(ground[1]), float(ground[2]), float(ground[3]),
                float(ground[4]), float(ground[5]), float(ground[6]))
    if boxa.get_iou(boxb) > thd:
        return True





