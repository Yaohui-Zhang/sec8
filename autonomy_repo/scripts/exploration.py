#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid
from asl_tb3_msgs.msg import TurtleBotState
import numpy as np
from scipy.signal import convolve2d
from asl_tb3_lib.grids import StochOccupancyGrid2D,snap_to_grid

class FrontierExploration(Node):
    def __init__(self):
        super().__init__('frontier_exploration')
        print("init")
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.goal_pub = self.create_publisher(TurtleBotState, '/cmd_nav', 10)
        self.nav_success_sub = self.create_subscription(Bool, "/nav_success",self.nav_callback, 10)
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.occupancy=None
        self.state=None
        #print(self.state)

    def nav_callback(self,msg:Bool)->None:
        if msg.data:
            current_state = self.occupancy.state2grid(np.array([self.state.x, self.state.y]))
            current_state = np.array([current_state[0],current_state[1]])
            frontier_states = []
            kernel=np.ones((13, 13))
            coverage = convolve2d(np.ones((self.occupancy.size_xy[1], self.occupancy.size_xy[0])), kernel, mode='same', boundary='fill', fillvalue=0)
            unknown=convolve2d(self.occupancy.probs==-1, kernel, mode='same', boundary='fill', fillvalue=0)
            known_occupied=convolve2d(self.occupancy.probs>=0.5, kernel, mode='same', boundary='fill', fillvalue=0)
            known_unoccupied=convolve2d((self.occupancy.probs >= 0) & (self.occupancy.probs < 0.5), kernel, mode='same', boundary='fill', fillvalue=0)
            unknown_smoothed=unknown/coverage
            known_occupied_smoothed=known_occupied/coverage
            known_unoccupied_smoothed=known_unoccupied/coverage
            criteria_1 = unknown_smoothed >= 0.2
            criteria_2 = known_occupied_smoothed == 0
            criteria_3 = known_unoccupied_smoothed >= 0.3
            criteria = criteria_1 & criteria_2 & criteria_3
            frontier_states = np.argwhere(criteria)
            if len(frontier_states)==0:
            	return
            state_positions = frontier_states[:, [1, 0]]
            distances = np.linalg.norm(state_positions - current_state, axis=1)
            min_dist_index = np.argmin(distances)
            new_msg=TurtleBotState()
            new_state=self.occupancy.grid2state(state_positions[min_dist_index])
            new_msg.x=new_state[0]
            new_msg.y=new_state[1]
            self.get_logger().warn(f"{new_state[0]},{new_state[1]}")
            self.goal_pub.publish(new_msg)
        else:
            return
            
    def state_callback(self, msg: TurtleBotState) -> None:
        """ callback triggered when receiving latest turtlebot state

        Args:
            msg (TurtleBotState): latest turtlebot state
        """
        self.state = msg


    def map_callback(self, msg: OccupancyGrid) -> None:
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )
        #print(occupancy.probs)

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExploration()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
