# Camera Lidar Fusion
This guide provides an overview of the camera and LiDAR fusion process used in our Formula Student car's perception system. By combining data from both sensors, we gain a more robust and detailed understanding of the surrounding environment, crucial for cone detection.
## Fusion Process
### Synchronize the inputs
We synchronize the camera and lidar inputs so that it can be fused together. In the current implementation we use "ApproximateTimeSynchronizer" from the "message_filters" package in ROS2. This calls a callback function when every it gets inputs from all the input sources.

### Filter the Point Cloud
The part of the point cloud which is not relevant is removed, This includes all the points which falls on the car body and all the points which are behind the car.

### Remove the Ground Plane
The ground plane is a flat surface that approximates the ground level in the 3D space represented by the point cloud data. It's essentially a virtual "floor" upon which objects like buildings, trees, or vehicles might be standing. All the points falling on it is removed so that the other obstacles can be isolated. In the current implementation it is done by using the ransac function implemented in Open3D. 

**Note:** A home-grown converter for converting the ROS2 Pointcloud to Open3D point cloud is used, the guide for using the converter can be found [here](https://github.com/NemB0t/ROS2-Open3D-PointCloud-Converter). The converter was used due to the limited implementation of the Pointcloud2 data structure in ROS2 Foxy and Galactic. This can be overcome when using ROS2 Humble and later. The converter will work on all versions of ROS2 which has Pointcloud2 implemented.

### Transform Lidar Point Cloud to Camera plane
Here we multiply all the lidar points with a transformation matrix to obtain a new set of points which has been transformed to the camera plane. This step is crucial since this ensures the points have proper values when projected on to the camera frame. The value of the transformation matrix is can be found using transforms or from the CAD model of the mount design. The current configured values are of the ADS_DV found in the EUFS simulator, Please calibrate it when using in real life. The actual calibration for the 2024 mount is in a previous commit of the same repo.

### Projecting the Lidar Points on to the camera frame
The lidar points are multiplied with the intrinsic camera matrix and the resulting points will be the 2D points on the camera plane.Read [this](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) article to get better understanding. The intrinsic matrix of the camera can be found by calibrating the camera but Zed wrapper provides this information in their camera info topic. 

**Note:** The points can be projected to 3D space using matrix projection as well but the results was losing the  depth information. To avoid loosing this information the data was stored in the "cloud_map" hash table when the points were transformed.

## Cone Detection
The cone is detected by using a YOLOV7-tiny model and the bounding boxes are returned. This module was only responsible for fusing the data. For details on training the model, checkout [depth_viewer](https://gitlab.com/uh4662410/uhra/fs-ai-perception/-/tree/ADS_DV_24/depth_viewer) module as this is out of scope for this guide.

## Depth Estimation
The average of all the points inside the bounding boxes are calculated and this data is published as the depth information of the detected cones. This is the most accurate way to estimate depth as stated in [this](https://www.mdpi.com/2073-8994/12/2/324) paper.

# Issues and Potential Solutions
1. The ApproximateTimeSynchronizer calls the call back at the lowest speed of the input. During testing we noticed that the camera input was very slow. This slowed down the output of the node. I was later informed that this can be increased by changing certain configuration files in the zed_wrapper.
2. The coordinates in the gazebo and the lidar point cloud were different in order to fix it, a permutation (x, y, z -> y, z, x) was performed. The idea was from KhalidOwlWalid's comment from  [this](https://answers.ros.org/question/393979/projecting-3d-points-into-pixel-using-image_geometrypinholecameramodel/) blog post.
3. The overall processing of the node can be boosted by rewriting the node in C++ and using faster data structures from STL for storing data.

Feel free to contact me for any doubts regarding the module. Good luck and all the best.