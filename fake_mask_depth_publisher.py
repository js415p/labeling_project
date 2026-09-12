#!/usr/bin/env python3
"""
fake_mask_depth_publisher.py

depth_map_projector 노드를 실제 카메라/모델 없이 테스트하기 위한
합성 depth/mask/camera_info publisher.

화면(640x480) 하단 중앙에 사각형 obstacle을 만들고, 그 영역의 depth를
1.2m로 고정해서 publish합니다. test_static_tf_publisher.py가 정의한
카메라 설치 위치(base_link 기준 앞 0.1m, 위 0.3m, 15도 숙임)를 알고 있으면,
map 좌표로 변환된 점들이 로버 앞 대략 1m 안팎, 바닥(z≈0) 근처에 찍히는지로
좌표 변환이 맞는지 눈으로 검증할 수 있습니다.

실행:
    python3 fake_mask_depth_publisher.py --ros-args \
        -p depth_topic:=/test/depth \
        -p mask_topic:=/test/mask \
        -p camera_info_topic:=/test/camera_info \
        -p frame_id:=camera_color_optical_frame
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class FakeMaskDepthPublisher(Node):
    def __init__(self):
        super().__init__('fake_mask_depth_publisher')

        self.declare_parameter('depth_topic', '/test/depth')
        self.declare_parameter('mask_topic', '/test/mask')
        self.declare_parameter('camera_info_topic', '/test/camera_info')
        self.declare_parameter('frame_id', 'camera_color_optical_frame')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fx', 525.0)
        self.declare_parameter('fy', 525.0)
        self.declare_parameter('obstacle_depth_m', 0.5)   # 장애물까지 거리 (m). 0.5m일 때 test_static_tf_publisher의
                                                            # 카메라 설치값(높이0.3m, 15도 숙임)과 조합하면 map/base_link
                                                            # 좌표 기준 z가 대략 0(바닥) 근처로 나오도록 미리 계산해둔 값.
        self.declare_parameter('publish_rate_hz', 5.0)

        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.frame_id = self.get_parameter('frame_id').value
        self.obstacle_depth_m = self.get_parameter('obstacle_depth_m').value

        self.bridge = CvBridge()

        self.depth_pub = self.create_publisher(Image, self.get_parameter('depth_topic').value, 10)
        self.mask_pub = self.create_publisher(Image, self.get_parameter('mask_topic').value, 10)
        self.info_pub = self.create_publisher(CameraInfo, self.get_parameter('camera_info_topic').value, 10)

        # 하단 중앙에 사각형 obstacle 영역 정의 (바닥에 붙은 물체를 흉내)
        self.obs_y0, self.obs_y1 = int(self.height * 0.75), int(self.height * 0.95)
        self.obs_x0, self.obs_x1 = int(self.width * 0.4), int(self.width * 0.6)

        rate = self.get_parameter('publish_rate_hz').value
        self.timer = self.create_timer(1.0 / rate, self.publish_frame)

        self.get_logger().info(
            f'fake publisher 시작. obstacle depth={self.obstacle_depth_m}m, '
            f'영역=({self.obs_x0}:{self.obs_x1}, {self.obs_y0}:{self.obs_y1}), frame_id={self.frame_id}')
        self.get_logger().info(
            '참고: test_static_tf_publisher.py 기본 설치값(카메라 높이 0.3m, 15도 숙임) 기준으로 '
            'obstacle_depth_m=0.5일 때 map/base_link 좌표계에서 예상 결과는 대략 '
            'x≈0.54m, y≈0m, z≈0.02m (바닥 근처) 입니다. depth_map_projector 출력이 이 근처로 '
            '나오면 좌표 변환 로직이 정상 동작하는 것으로 볼 수 있습니다.')

    def publish_frame(self):
        now = self.get_clock().now().to_msg()

        # ---- depth (16UC1, mm 단위, RealSense와 동일 컨벤션) ----
        depth = np.zeros((self.height, self.width), dtype=np.uint16)
        depth_mm = int(self.obstacle_depth_m * 1000)
        depth[self.obs_y0:self.obs_y1, self.obs_x0:self.obs_x1] = depth_mm
        # 배경도 완전히 0이면 노이즈처럼 보이니 먼 배경(4m)로 채움
        background_mask = depth == 0
        depth[background_mask] = 4000

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='16UC1')
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = self.frame_id
        self.depth_pub.publish(depth_msg)

        # ---- mask (mono8, obstacle=255) ----
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        mask[self.obs_y0:self.obs_y1, self.obs_x0:self.obs_x1] = 255
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header.stamp = now
        mask_msg.header.frame_id = self.frame_id
        self.mask_pub.publish(mask_msg)

        # ---- camera_info ----
        info = CameraInfo()
        info.header.stamp = now
        info.header.frame_id = self.frame_id
        info.width = self.width
        info.height = self.height
        info.k = [self.fx, 0.0, self.cx,
                  0.0, self.fy, self.cy,
                  0.0, 0.0, 1.0]
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.info_pub.publish(info)


def main(args=None):
    rclpy.init(args=args)
    node = FakeMaskDepthPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
