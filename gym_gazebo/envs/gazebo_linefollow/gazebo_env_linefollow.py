import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np
from functools import reduce

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected

    def split_img(self, image, splits):
        '''
            @brief Splits an image into splits number of images along width 
            @param image : Image to be spliced

            @retval (imgs)
        '''
        width = image.shape[1]
        segment_width = width // splits

        imgs = []

        for i in range(splits):
            start_col = i * segment_width
            end_col = (i + 1) * segment_width if i < splits - 1 else width
        
            segment = image[:, start_col:end_col]
            imgs.append(segment)

        return imgs

    def process_image(self, data):
        '''
            @brief Converts ROS image to OpenCV, analyzes last third for line,
                draws sections and detected line, and returns state & done
            @param data : Image data from ROS
            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return [0]*10, True

        num_split = 10
        height, width = cv_image.shape[:2]
        light_thresh = 100  # threshold for detecting dark line

        gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # roi: last third of the image
        last_third = gray_img[2*height//3:, :]

        #split into 10 sec using my function
        split_img_arr = self.split_img(last_third, num_split)

        region_pixels = {}  # section : number of dark pixels
        non_line_sections = 0

        for section, img_section in enumerate(split_img_arr):
            #count dark pixels in this section
            line_pixels = np.sum(img_section < light_thresh)
            region_pixels[section] = line_pixels
            if line_pixels == 0:
                non_line_sections += 1

        # Determine state
        if non_line_sections == num_split:
            self.timeout += 1
            state = [0] * num_split
            max_section = -1
        else:
            self.timeout = 0
            # find section with max line pixels
            max_section = max(region_pixels, key=region_pixels.get)
            state = [0] * num_split
            state[max_section] = 1

        done = False
        if self.timeout >= 30:
            done = True
            self.timeout = 0

        # ---------------------- Visualization ----------------------
        vis_image = cv_image.copy()
        section_width = width // num_split
        start_y = 2*height//3

        # Draw vertical section lines
        for i in range(1, num_split):
            x = i * section_width
            cv2.line(vis_image, (x, start_y), (x, height), (255, 0, 0), 2)

        # Draw dot at detected section
        if max_section >= 0:
            dot_x = max_section * section_width + section_width // 2
            dot_y = start_y + (height//6)  # middle of last third
            cv2.circle(vis_image, (dot_x, dot_y), 5, (0, 0, 255), -1)

        cv2.imshow("Line Detection", vis_image)
        cv2.waitKey(1)

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        if not done:
            #old reward scheme
            if action == 0:  # FORWARD
               reward = 4
            elif action == 1:  # LEFT
               reward = 2
            else:
               reward = 2  # RIGHT
        else:
            reward = -200


        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state