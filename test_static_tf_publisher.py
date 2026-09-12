#!/usr/bin/env python3
"""
test_static_tf_publisher.py

실제 로버 없이 depth_map_projector 노드를 독립적으로 테스트하기 위한
가짜 TF 트리 broadcaster.

구성: map -> odom -> base_link -> camera_link -> camera_color_optical_frame

가정:
- map == odom == base_link 위치 (로버가 map 원점에 정지해 있다고 가정)
- camera_link: base_link 기준 앞으로 10cm, 위로 30cm에 설치, 바닥 쪽으로 15도 숙임
- camera_color_optical_frame: camera_link 기준 표준 광학 좌표계 변환
  (REP-103/105: optical z=forward, x=right, y=down)

실제 로버에서는 이 노드 대신 robot_state_publisher(URDF) + SLAM/AMCL이
이 TF들을 채워줍니다. 이건 어디까지나 좌표 변환 로직 자체를 검증하기 위한
임시 스텁입니다.
"""

import math
import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped


def quat_from_rpy(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy,  # w
    )


class TestStaticTfPublisher(Node):
    def __init__(self):
        super().__init__('test_static_tf_publisher')
        self.broadcaster = StaticTransformBroadcaster(self)

        transforms = []

        # map -> odom (identity)
        transforms.append(self.make_tf('map', 'odom', 0, 0, 0, 0, 0, 0, 1))
        # odom -> base_link (identity, 로버가 원점에 정지)
        transforms.append(self.make_tf('odom', 'base_link', 0, 0, 0, 0, 0, 0, 1))

        # base_link -> camera_link : 앞으로 0.1m, 위로 0.3m, pitch -15도(바닥 쪽으로 숙임)
        qx, qy, qz, qw = quat_from_rpy(0.0, math.radians(15), 0.0)
        transforms.append(self.make_tf('base_link', 'camera_link', 0.1, 0.0, 0.3, qx, qy, qz, qw))

        # camera_link -> camera_color_optical_frame : 표준 카메라 광학 좌표계 변환
        # (robot: x-forward,y-left,z-up)  ->  (optical: x-right,y-down,z-forward)
        qx, qy, qz, qw = (-0.5, 0.5, -0.5, 0.5)
        transforms.append(self.make_tf('camera_link', 'camera_color_optical_frame', 0, 0, 0, qx, qy, qz, qw))

        self.broadcaster.sendTransform(transforms)
        self.get_logger().info('테스트용 static TF 발행 완료: map -> odom -> base_link -> camera_link -> camera_color_optical_frame')

    def make_tf(self, parent, child, x, y, z, qx, qy, qz, qw) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)
        return t


def main(args=None):
    rclpy.init(args=args)
    node = TestStaticTfPublisher()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
