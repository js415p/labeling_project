#!/usr/bin/env python3
"""
debug_pointcloud_echo.py

/vision_obstacles (PointCloud2)를 구독해서 실제 x,y,z 좌표값을 콘솔에 찍어주는
디버그용 스크립트. RViz는 대략적인 위치만 보여주므로, 정확한 숫자 비교가 필요할 때 사용.

실행:
    python3 debug_pointcloud_echo.py --ros-args -p topic:=/vision_obstacles
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class DebugPointCloudEcho(Node):
    def __init__(self):
        super().__init__('debug_pointcloud_echo')
        self.declare_parameter('topic', '/vision_obstacles')
        topic = self.get_parameter('topic').value
        self.sub = self.create_subscription(PointCloud2, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f'{topic} 구독 시작. 메시지 들어오면 앞 5개 포인트 좌표를 출력합니다.')

    def cb(self, msg: PointCloud2):
        points = list(point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))
        self.get_logger().info(f'frame_id={msg.header.frame_id}, 총 포인트 수={len(points)}')
        for i, p in enumerate(points[:5]):
            self.get_logger().info(f'  point[{i}] = x={p[0]:.3f}, y={p[1]:.3f}, z={p[2]:.3f}')


def main(args=None):
    rclpy.init(args=args)
    node = DebugPointCloudEcho()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
