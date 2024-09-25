import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs.msg import PointCloud2, PointField
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from .converter import Converter
import cv2
from cv_bridge import CvBridge, CvBridgeError
from os import getcwd
import onnxruntime as ort
from eufs_msgs.msg import BoundingBoxes, BoundingBox, ConeWithCovariance, ConeArrayWithCovariance
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header, ColorRGBA
from message_filters import ApproximateTimeSynchronizer,Subscriber


class c_l_fusion(Node):

    def __init__(self, cuda=True):
        super().__init__('minimal_subscriber')
        self.cam_height = None
        self.cam_width = None

        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.lidar_sub=Subscriber(self,PointCloud2,'/velodyne_points',qos_profile=qos_profile)
        self.left_cam_img_sub=Subscriber(self,Image,'/zed/left/image_rect_color',qos_profile=qos_profile)
        self.left_cam_info_sub=Subscriber(self,CameraInfo,'/zed/left/camera_info',qos_profile=qos_profile)

        # Create the approximate time synchroniser to sync the subscribers
        self.ats_ = ApproximateTimeSynchronizer(
            [self.lidar_sub, self.left_cam_img_sub, self.left_cam_info_sub],
            100,
            0.5  # The slop should almost always be small!
        )
        # # Register the callback with the approximate time synchroniser
        self.ats_.registerCallback(self.callback)


        # Variables for the yoloV7 cone detector from depthviewer.py
        self.onnx_path = getcwd() + "/src/perception/cam_lidar_fusion/cam_lidar_fusion/model/prev_best.onnx"
        self.provider = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.onnx_path, providers=self.provider)
        self.names = ['blue_cone', 'large_orange_cone', 'orange_cone', 'yellow_cone']
        self.publisher = self.create_publisher(ConeArrayWithCovariance, '/cones', 10)
        self.cone_marker_publisher = self.create_publisher(MarkerArray, 'cone_markers', 10)

        self.lidar_PC_publisher = self.create_publisher(PointCloud2, '/FilteredPointCloud', 10)

    def callback(self,cloud,left_image,left_image_info):
        converter = Converter()
        left_image = np.asarray(self.bridge.imgmsg_to_cv2(left_image, "bgr8"))
        cloud = converter.ROSpc2_to_nparray(cloud)[0]
        cloud = self.filter_pointcloud(cloud)
        cloud = self.remove_ground_plane(cloud, converter)
        og_cloud = cloud
        cloud = self.transform_lidar_points_to_camera_frame(cloud)
        #TODO: Method to test transformed lidar points on the camera plane
        # self.lidar_PC_publisher.publish(converter.nparray_to_ROSpc2(cloud, "zed_left_camera_optical_frame"))  # zed_left_camera_optical_frame

        left_intrinsic_camera_matrix=np.asarray(left_image_info.k).reshape(3, 3)
        cam_width = left_image_info.width
        cam_height = left_image_info.height
        cones = set()#detected cones will be in this set
        permutation_matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])  # Fixing the axis,Thx to KhalidOwlWalid from https://answers.ros.org/question/393979/projecting-3d-points-into-pixel-using-image_geometrypinholecameramodel/
        cloud = permutation_matrix @ cloud.T  # (x, y, z) to (y, z, x) this is done to fix the coordinate issue with ros and gazebo
        cloud = cloud.T
        # converting the 3D lidar to 2D pixels and storing it into a cloud_map (hashmap) where key= pixels in (u,v) format and value = 3d lidar points
        # cloud_map will be used to get the 3D values from (u,v)
        cloud_map=self.project_points_and_generate_cloudmap(cloud,left_intrinsic_camera_matrix,og_cloud,cam_width,cam_height)#This is used to visalise the lidar points(cloud,left_intrinsic_camera_matrix,og_cloud,cam_width,cam_height,left_image)
        # self.get_logger().info(str(cloud_map))

        ##Detect cones using Yolov7
        img = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        bboxes_left = self.detect_cones(img)

        img_points = np.array([[x, y] for x, y in cloud_map.keys()])
        for bbox_left in bboxes_left:
            xmin, ymin, xmax, ymax = int(bbox_left.xmin), int(bbox_left.ymin), int(bbox_left.xmax), int(bbox_left.ymax)
            color, probability = bbox_left.color, bbox_left.probability
            ##averaging depths of all points in bounding box             # Idea is from https://www.mdpi.com/2073-8994/12/2/324
            indices = self.points_in_bbox(img_points, xmin, ymin, xmax, ymax)
            # selected points inside the bounding box
            # self.get_logger().info(str(img_points))
            img_points = img_points[indices]
            if len(img_points) < 1:
                continue
            avg_x = avg_y = 0
            for u, v in img_points:
                # TODO Uncomment to visualize correlated lidar point
                # if (u,v) in cloud_map:
                #             cv2.circle(left_image, (int(u), int(v)), 2, (0, 255, 0) )
                x, y, _ = cloud_map[(u, v)]
                avg_y += y
                avg_x += x
            avg_y = avg_y / len(img_points)
            avg_x = avg_x / len(img_points)
            x, y = avg_x, avg_y
            point_precision = 3
            x, y = round(x, point_precision), round(y, point_precision)
            cones.add((x, y, color))
            img_points = np.array([[x, y] for x, y in cloud_map.keys()])
            # TODO Uncomment to visualize detected cone bounding boxes
            # cv2.rectangle(left_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1) # Detected cones
        # Publish /cones topic containing all cones with color
        self.publish_cones(cones)
        # self.get_logger().info(str(self.cones))

        # TODO Uncomment to visualize
        # cv2.imshow('test', left_image)
        # cv2.waitKey(1)

    def project_points_and_generate_cloudmap(self, points_homog, camera_matrix,og_cloud,cam_width,cam_height,img=[]): #maps 3d->2d
        points_homog=np.asarray(points_homog)
        # Project points using camera matrix
        camera_matrix=camera_matrix @ self.rotation_z(180)
        # camera_matrix = camera_matrix @ self.rotation_y(180)
        image_points_homog = camera_matrix @ points_homog.T

        # Normalize homogeneous coordinates and extract u, v
        image_points_norm = image_points_homog / image_points_homog[2]  # Divide by w
        u = image_points_norm[0]
        v = image_points_norm[1]

        #filtering the points which are inside the image
        uv_index= (u >= 0) & (u <= cam_width) & (v>=0) &  (v<=cam_height)
        # self.get_logger().info(str(len(u))+ ' '+str(len(v))+' '+str(len(og_cloud)))
        u=u[uv_index]
        v=v[uv_index]
        og_cloud=og_cloud[uv_index]

        # TODO Uncomment to visualize mapped lidar points in the camera frame #Note the visualization is very slow
        # for i in range(len(u)):
        #     yellow = (0, 255, 255)
        #     cv2.circle(img, (int(u[i]), int(v[i])), 1, yellow)
        # cv2.imshow('test', img)
        # cv2.waitKey(1)

        # Ensure u, v, and points_homog are aligned correctly
        assert len(u) == len(v) == len(og_cloud)

        # Create an array of tuples representing the 2D points
        uv_tuples = np.column_stack((u, v)).astype(int)

        # Convert the array of tuples to a list of tuples (required for creating a dictionary)
        uv_tuples_list = [tuple(uv) for uv in uv_tuples]
        # uv_tuples_list = list(map(tuple, uv_tuples)) # incase you face performance issue use this

        # Create the dictionary using a dictionary comprehension
        cloud_map = dict(zip(uv_tuples_list, og_cloud)) # Hash map to store (u,v) to [x,y,z] mapping - 2d to 3d mapping

        return cloud_map#np.stack((u, v)).T,cloud_map

    def transform_lidar_points_to_camera_frame(self, cloud): #Optimised
        #TODO The translation and rotaion matrix needs to be calibrated based on real car or using cad
        # Define the fixed transformation components
        translation_matrix = np.array([-0.097, -0.060, 0.050]).reshape(3, 1)
        q = np.array([0.000, -0.035, 0.000, 0.999])
        # rotation_matrix = self.quaternion_rotation_matrix(q) @ self.rotation_z(180)
        rotation_matrix = self.rotation_y(360-3)

        # Create the fixed transformation matrix
        transformation_matrix = np.hstack([rotation_matrix, translation_matrix])
        transformation_matrix = np.vstack([transformation_matrix, [0, 0, 0, 1]])

        # Apply the transformation
        cloud_homogeneous = np.hstack([cloud, np.ones((cloud.shape[0], 1))])
        cloud_transformed = transformation_matrix @ cloud_homogeneous.T

        # Convert back to non-homogeneous coordinates
        cloud_transformed = cloud_transformed[:3, :].T

        return cloud_transformed

    def remove_ground_plane(self, cloud, converter):#optimised
        # Convert point cloud from numpy array to Open3D point cloud
        o3Dpc = converter.nparray_to_O3DPointCloud(cloud)

        # Perform RANSAC to find the ground plane
        seed_indices = self.random_sampling(o3Dpc)
        ransac_input = o3Dpc.select_by_index(seed_indices)
        plane_model, _ = ransac_input.segment_plane(
            distance_threshold=0.1,
            ransac_n=3,
            num_iterations=1000,
        )

        # Calculate the distance of each point from the plane
        plane_normal = np.array(plane_model[:3])
        plane_offset = plane_model[3]
        distances = np.abs(np.dot(cloud[:, :3], plane_normal) + plane_offset)
        distances /= np.linalg.norm(plane_normal)

        # Filter out ground points
        ground_threshold = 0.1
        non_ground_mask = distances >= ground_threshold
        non_ground_plane_pts = cloud[non_ground_mask]

        # self.lidar_PC_publisher.publish(converter.nparray_to_ROSpc2(non_ground_plane_pts,"velodyne"))
        # Return the filtered points
        return non_ground_plane_pts

    def quaternion_rotation_matrix(self,Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                 This rotation matrix converts a point in the local reference
                 frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                               [r10, r11, r12],
                               [r20, r21, r22]])

        return rot_matrix

    def rotation_z(self,angle):
        """
        Generates a rotation matrix for rotation around the Z-axis.

        Args:
            angle: Rotation angle (in radians).

        Returns:
            A numpy array (3, 3) representing the rotation matrix.
        """
        # Convert angle to radians (if needed)
        angle_rad = np.radians(angle)

        # Rotation matrix for Z-axis rotation
        Rz = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                       [np.sin(angle_rad), np.cos(angle_rad), 0],
                       [0, 0, 1]])

        return Rz

    def rotation_y(self,angle):
        # Convert angle to radians
        angle = np.radians(angle)

        return np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])

    def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        #image preprocessor function from depth_viewer.py
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        scale_ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            scale_ratio = min(scale_ratio, 1.0)

        new_unpad = int(round(shape[1] * scale_ratio)), int(round(shape[0] * scale_ratio))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        # divide padding into 2 sides
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        # add border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img, scale_ratio, (dw, dh)

    def detect_cones(self, image):
        # Cone detection funtion from depthviewer.py
        # Implement your cone detection logic here using a pre-trained model
        # Return a list of bounding boxes with class names, probabilities, and coordinates
        # in the format of the BoundingBoxes message

        # self.get_logger().info('Subscribed Image from Zed2i')
        # try:
        #     img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # except CvBridgeError as e:
        #     self.get_logger().info(e)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image = img.copy()
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255

        outname = [i.name for i in self.session.get_outputs()]
        inname = [i.name for i in self.session.get_inputs()]
        inp = {inname[0]: im}
        out = self.session.run(outname, inp)[0]
        out_msg = BoundingBoxes()

        for (batch_id, x0, y0, x1, y1, cls_id, score) in out:
            if score > 0.5:
                # self.get_logger().info("str(cls_id): -------  ")
                self.get_logger().info(str(self.names[int(cls_id)]))
                bbox = BoundingBox()
                box = np.array([x0, y0, x1, y1])
                box -= np.array(dwdh * 2)
                box /= ratio
                box = box.round().astype(np.int32).tolist()
                name = self.names[int(cls_id)]
                bbox.xmin = max(float(box[0]), 0.0)
                bbox.ymin = max(float(box[1]), 0.0)
                bbox.xmax = max(float(box[2]), 0.0)
                bbox.ymax = max(float(box[3]), 0.0)
                # bbox.color = str(self.colors[name])
                bbox.color = str(self.names[int(cls_id)])

                # self.get_logger().info("self.names:----------------")
                # self.get_logger().info(str(self.names))
                # self.get_logger().info("self.colors:----------------")
                # self.get_logger().info(str(self.colors ))
                # self.get_logger().info("self.names[int(cls_id):----------------")
                # self.get_logger().info(str(self.names[int(cls_id)]))
                # bbox.color = "orange"

                # bbox.class_id = self.names[int(cls_id)]
                bbox.probability = float(score)
                self.get_logger().info(str(score))

                out_msg.bounding_boxes.append(bbox)
        # print(out_msg)
        # self.get_logger().info("Out Message")
        # self.get_logger().info(str(out_msg))
        # self.bboxes.append(bbox)
        # self.pub.publish(out_msg)
        # self.processed_left_image_pub = self.create_publisher(Image, "/DepthViewer/Processed_left_image", 1)
        return out_msg.bounding_boxes

    def publish_cones(self, cones):
        cone_array_msg = ConeArrayWithCovariance()
        cone_array_msg.header = Header()
        cone_array_msg.header.stamp = self.get_clock().now().to_msg()

        # color_map = {
        #         'blue': 'blue_cone',
        #         'yellow': 'yellow_cone',
        #         'orange': 'orange_cone',
        #         'big_orange': 'big_orange_cone'
        #     }

        # Populate the cones based on the provided list
        for cone in cones:
            cone_msg = ConeWithCovariance()
            # cone_msg.point = Point(float(x=cone[0]), float(y=cone[1]), z=0.0)  # Assuming z-coordinate is 0.0 for simplicity
            cone_msg.point = Point()
            cone_msg.point.x = float(cone[0])
            cone_msg.point.y = float(cone[1])
            cone_msg.point.z = 0.0  # Assuming z-coordinate is 0.0 for simplicity

            # self.get_logger().info("cone=====================")
            # self.get_logger().info(str(cone))

            # # Add the cone message to the appropriate color list
            # color = cone[2]
            # if color in color_map:
            #     getattr(cone_array_msg, color_map[color]).append(cone_msg)

            # Add the cone message to the appropriate color list
            color = cone[2]
            if color == 'blue_cone':
                cone_array_msg.blue_cones.append(cone_msg)
            elif color == 'yellow_cone':
                cone_array_msg.yellow_cones.append(cone_msg)
            elif color == 'orange_cone':
                cone_array_msg.orange_cones.append(cone_msg)
            elif color == 'large_orange_cone':
                cone_array_msg.big_orange_cones.append(cone_msg)
            else:
                cone_array_msg.unknown_color_cones.append(cone_msg)

            # # getattr(cone_array_msg, f"{cone[2]}_cones").append(cone_msg)

        self.publisher.publish(cone_array_msg)
        self.publish_detected_cones(cones)

    def publish_detected_cones(self, cones):
        marker_array_msg = MarkerArray()

        # Create a delete marker with action DELETEALL
        delete_marker = Marker()
        delete_marker.header.frame_id = "base_link"
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = "cones"
        delete_marker.action = Marker.DELETEALL

        # Add the delete marker to the marker array
        marker_array_msg.markers.append(delete_marker)

        # Publish the marker array to clear the old markers
        self.cone_marker_publisher.publish(marker_array_msg)

        for i, cone in enumerate(cones):
            color = (0, 255, 0)
            label = cone[2]

            if label == 'blue_cone':
                color = (255, 0, 0)  # Red for blue cones
            elif label == 'orange_cone':
                color = (0, 165, 255)  # Orange for orange cones
            elif label == 'yellow_cone':
                color = (0, 255, 255)
            elif label == 'large_orange_cone':
                color = (0, 165, 255)

            marker = Marker()
            marker.header.frame_id = 'base_footprint'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'cone_markers'
            marker.id = i
            marker.type = Marker.CYLINDER
            # marker.type = Marker.
            marker.action = Marker.ADD
            marker.pose.position.x = float(cone[0])
            marker.pose.position.y = float(cone[1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.5  # Cone diameter
            marker.scale.y = 0.5  # Cone diameter
            marker.scale.z = 0.75  # Cone height
            # marker.color = ColorRGBA(*[c / 255.0 for c in color])
            # marker.color.a = 1.0
            # marker.color.r = 1.0  # Set marker color based on cone color
            # marker.color.g = 0.0
            # marker.color.b = 0.0
            marker.color.r = color[2] / 255.0  # Blue channel
            marker.color.g = color[1] / 255.0  # Green channel
            marker.color.b = color[0] / 255.0  # Red channel
            marker.color.a = 1.0

            marker_array_msg.markers.append(marker)

        self.cone_marker_publisher.publish(marker_array_msg)

    def points_in_bbox(self,points, xmin, ymin, xmax, ymax):
        """
        Finds all points within the bounding box defined by (xmin, ymin, xmax, ymax).

        Args:
            points: A NumPy array of shape (N, 2) where each row represents a point (x, y).
            xmin: Minimum x-coordinate of the bounding box.
            ymin: Minimum y-coordinate of the bounding box.
            xmax: Maximum x-coordinate of the bounding box.
            ymax: Maximum y-coordinate of the bounding box.

        Returns:
            A NumPy array of indices for points within the bounding box.
        """
        # Efficiently select points within bounding box using boolean indexing
        return np.where((points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
                        (points[:, 1] >= ymin) & (points[:, 1] <= ymax))[0]

    def random_sampling(self, pcd_o3d):
        """Samples points from the point cloud randomly

        Args:
            pcd_o3d (o3d.geometry.PointCloud): Point cloud data in Open3D format

        Returns:
            List[int]: Indices of the sampled points
        """
        sampled_point_indices = np.random.choice(
            len(pcd_o3d.points),
            200,
            replace=False,
        )

        return sampled_point_indices

    def filter_pointcloud(self,cloud):#Optimized
        ##Filtering lidar points which are not in camera frame
        # Removeing points which are behind the camera
        cloud = cloud[(cloud[:, 0] > 0.094)]  # x value of translation matrix
        # cloud = cloud[(cloud[:, 0] < 5)]  # For testing
        # cloud = cloud[(cloud[:, 1] <0)]  # For testing

        # Removing points which are falling on the car
        #TODO Adjust this value based on testing to filter out the lidar points falling on the entire length of the car
        cloud = cloud[np.linalg.norm(cloud, axis=1) > 1]  # Calculate distances from origin for all points at once
        return cloud
def main(args=None):
    rclpy.init(args=args)
    c_l_subscriber = c_l_fusion()
    rclpy.spin(c_l_subscriber)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    c_l_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()