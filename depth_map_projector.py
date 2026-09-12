#!/usr/bin/env python3
"""
depth_map_projector.py

obstacle mask(mono8) + depth image + camera_info 를 입력받아
mask==obstacle 인 픽셀들의 3D 좌표를 계산하고, tf2로 target_frame(기본 map)까지
변환하여 PointCloud2로 publish 하는 ROS2 prototype 노드.

실행 예:
    python3 depth_map_projector.py --ros-args \
        -p mask_topic:=/obstacle_mask \
        -p depth_topic:=/camera/aligned_depth_to_color/image_raw \
        -p camera_info_topic:=/camera/aligned_depth_to_color/camera_info \
        -p target_frame:=map \
        -p depth_scale:=0.001 \
        -p point_stride:=4

RViz에서 확인하는 법:
    1. Fixed Frame을 target_frame과 동일하게 (기본 map)
    2. PointCloud2 display 추가, Topic을 output_topic(기본 /vision_obstacles)으로 설정
    3. 로버를 실제로 움직이면서 장애물 위치에 점들이 바닥에 맞게 찍히는지 확인

주의: mask와 depth는 반드시 "정렬(aligned)"되어 있어야 합니다.
      즉 같은 픽셀 좌표가 같은 3D 지점을 가리켜야 함 (RealSense라면
      aligned_depth_to_color 토픽 사용, 다른 카메라도 depth-to-rgb 정렬 필요).
"""

import struct

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge

import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


class DepthMapProjector(Node):
    def __init__(self):
        super().__init__('depth_map_projector')

        # ---- 파라미터 선언 (실제 로버 토픽/프레임에 맞게 실행 시 override) ----
        self.declare_parameter('mask_topic', '/obstacle_mask')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('output_topic', '/vision_obstacles')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('depth_scale', 0.001)   # RealSense mm->m 기본값. 카메라마다 다를 수 있음
        self.declare_parameter('min_depth', 0.1)        # m
        self.declare_parameter('max_depth', 5.0)         # m
        self.declare_parameter('point_stride', 4)        # 픽셀 몇 개당 1개 포인트 뽑을지 (다운샘플, 클라우드 크기 절약)
        self.declare_parameter('mask_threshold', 127)     # 이 값 이상이면 obstacle 픽셀로 간주

        self.mask_topic = self.get_parameter('mask_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.target_frame = self.get_parameter('target_frame').value
        self.depth_scale = self.get_parameter('depth_scale').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.point_stride = int(self.get_parameter('point_stride').value)
        self.mask_threshold = int(self.get_parameter('mask_threshold').value)

        self.bridge = CvBridge()

        # camera intrinsic은 camera_info에서 한 번만 받아서 캐싱 (보통 안 바뀜)
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_cb, qos_profile_sensor_data)

        # mask + depth 시간 동기화
        self.mask_sub = message_filters.Subscriber(self, Image, self.mask_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.mask_sub, self.depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.synced_cb)

        self.pub = self.create_publisher(PointCloud2, self.output_topic, 10)

        # tf2 buffer/listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f'depth_map_projector 시작. mask={self.mask_topic}, depth={self.depth_topic}, '
            f'camera_info={self.camera_info_topic}, target_frame={self.target_frame}, output={self.output_topic}')

    def camera_info_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def synced_cb(self, mask_msg: Image, depth_msg: Image):
        if self.fx is None:
            self.get_logger().warn('아직 camera_info를 못 받음, 이번 프레임 스킵', throttle_duration_sec=2.0)
            return

        mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        depth = depth.astype(np.float32) * self.depth_scale  # 미터 단위로 변환

        obstacle_ys, obstacle_xs = np.where(mask >= self.mask_threshold)
        if len(obstacle_xs) == 0:
            return  # 이 프레임엔 장애물 없음

        # 다운샘플 (성능/클라우드 크기 절약)
        if self.point_stride > 1:
            keep = np.arange(0, len(obstacle_xs), self.point_stride)
            obstacle_xs = obstacle_xs[keep]
            obstacle_ys = obstacle_ys[keep]

        z = depth[obstacle_ys, obstacle_xs]
        valid = (z > self.min_depth) & (z < self.max_depth) & np.isfinite(z)
        if not np.any(valid):
            return

        u = obstacle_xs[valid].astype(np.float32)
        v = obstacle_ys[valid].astype(np.float32)
        z = z[valid]

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        points_camera = np.stack([x, y, z], axis=-1)  # (N,3), camera optical frame 기준

        # camera optical frame 기준 PointCloud2 생성
        cloud_cam = self.make_pointcloud2(points_camera, depth_msg.header.frame_id, depth_msg.header.stamp)

        # target_frame으로 변환
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, depth_msg.header.frame_id, depth_msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.2))
        except Exception as e:
            self.get_logger().warn(f'TF lookup 실패 ({depth_msg.header.frame_id} -> {self.target_frame}): {e}',
                                    throttle_duration_sec=2.0)
            return

        cloud_target = do_transform_cloud(cloud_cam, transform)
        self.pub.publish(cloud_target)

    @staticmethod
    def make_pointcloud2(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = DepthMapProjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
