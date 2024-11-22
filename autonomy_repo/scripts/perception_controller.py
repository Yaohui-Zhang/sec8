#!/usr/bin/env python3

import rclpy                    # ROS2 client library
import numpy as np
import time

from asl_tb3_lib.control import BaseController
from asl_tb3_msgs.msg import TurtleBotControl
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
class PerceptionController(BaseController):
    def __init__(self) -> None:
        super().__init__("perception_controller")
        self.declare_parameter("active", True)
        self.detected_time = None

        self.create_subscription(Bool, "/detector_bool", self.detected_callback, 10)

    def detected_callback(self, msg: Bool):
        if msg.data == True:
            pre_time = self.detected_time
            self.detected_time = self.get_clock().now().nanoseconds / 1e9
            if pre_time is None:
                self.set_parameters([rclpy.Parameter("active", value=False)])
            else:
                if self.detected_time-pre_time>=6:
                    self.set_parameters([rclpy.Parameter("active", value=False)])
            # self.set_parameters([rclpy.Parameter("active", value=False)])

                
    
    @property
    def active(self)->bool:
        return self.get_parameter("active").value
 
    def compute_control(self) -> TurtleBotControl:
        control_msg = TurtleBotControl()
        if self.active:
            control_msg.omega = 0.5
        else:
            twist = Twist()
            twist.angular.z = 0.
            self.cmd_vel_pub.publish(twist)
            time.sleep(5.)
            self.set_parameters([rclpy.Parameter("active", value=True)])
        return control_msg


if __name__ == "__main__":
    rclpy.init()
    node = PerceptionController()
    rclpy.spin(node)
    rclpy.shutdown()
    