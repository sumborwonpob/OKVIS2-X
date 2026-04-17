/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
 * @file Subscriber.cpp
 * @brief Source file for the Subscriber class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */
 
#include <glog/logging.h>
#include <sstream>
#include <okvis/ros2/Subscriber.hpp>
#include <okvis/ros2/PointCloudUtilities.hpp>
#include <pcl_conversions/pcl_conversions.h>

#if USE_LIVOX_CUSTOM_MSG
#include <livox_ros_driver2/msg/custom_msg.hpp>
#endif

// Set to 1 to use compressed images, 0 to use uncompressed
#ifndef USE_COMPRESSED_IMAGE
#define USE_COMPRESSED_IMAGE 1
#endif
// Set to 1 to use Livox CustomMsg, 0 to use PointCloud2
#ifndef USE_LIVOX_CUSTOM_MSG
#define USE_LIVOX_CUSTOM_MSG 1
#endif
#ifdef OKVIS_USE_NN
#include <okvis/Processor.hpp>
#endif

constexpr double kDefaultSyncToleranceSec = 0.1; ///< Fallback sync threshold in seconds.

/// \brief okvis Main namespace of this package.
namespace okvis {

Subscriber::~Subscriber()
{
  if (imgTransport_ != nullptr)
    imgTransport_.reset();
}

Subscriber::Subscriber(std::shared_ptr<rclcpp::Node> node, 
                       okvis::ViInterface* viInterfacePtr,
                       okvis::Publisher* publisher, 
                       const okvis::ViParameters& parameters,
                       okvis::SubmappingInterface* seInterface,
                       bool isDepthCamera, bool isLiDAR)
{
  viInterface_ = viInterfacePtr;
  seInterface_ = seInterface;
  publisher_ = publisher;
  parameters_ = parameters;
  setNodeHandle(node, isDepthCamera, isLiDAR);
}

void Subscriber::setNodeHandle(std::shared_ptr<rclcpp::Node> node,
                               bool isDepthCamera, bool isLiDAR)
{
  node_ = node;

  imageSubscribers_.resize(parameters_.nCameraSystem.numCameras());
  compressedImageSubscribers_.resize(parameters_.nCameraSystem.numCameras());
  depthImageSubscribers_.resize(parameters_.nCameraSystem.numCameras());
  compressedDepthImageSubscribers_.resize(parameters_.nCameraSystem.numCameras());
  imagesReceived_.resize(parameters_.nCameraSystem.numCameras());
  depthImagesReceived_.resize(parameters_.nCameraSystem.numCameras());

  // set up image reception
  if (imgTransport_ != nullptr)
    imgTransport_.reset();
  imgTransport_ = std::make_shared<image_transport::ImageTransport>(node_);

  // set up callbacks
  for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
#if USE_COMPRESSED_IMAGE
    const size_t camIdx = i;
    const bool isColour = parameters_.nCameraSystem.cameraType(i).isColour;
    compressedImageSubscribers_[i] = node_->create_subscription<sensor_msgs::msg::CompressedImage>(
        "/okvis/cam" + std::to_string(i) + "/compressed",
        rclcpp::SensorDataQoS(),
        [this, camIdx, isColour](const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) {
          this->compressedImageCallback(msg, camIdx, isColour);
        });
#else
    imageSubscribers_[i] = imgTransport_->subscribe(
        "/okvis/cam" + std::to_string(i) +"/image_raw",
        30 * parameters_.nCameraSystem.numCameras(),
        std::bind(&Subscriber::imageCallback, this, std::placeholders::_1, i,
          parameters_.nCameraSystem.cameraType(i).isColour));
#endif
  }

  subImu_ = node_->create_subscription<sensor_msgs::msg::Imu>(
      "/okvis/imu0", rclcpp::SensorDataQoS(),
      std::bind(&Subscriber::imuCallback, this, std::placeholders::_1));

  if(isDepthCamera){
    syncDepthImages_ = true;

    // set up callbacks
    for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
      const size_t depthCamIdx = i;
      compressedDepthImageSubscribers_[i] = node_->create_subscription<sensor_msgs::msg::CompressedImage>(
        "/okvis/depth" + std::to_string(i) + "/compressed",
        30 * parameters_.nCameraSystem.numCameras(),
        [this, depthCamIdx](const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) {
          this->compressedDepthCallback(msg, depthCamIdx);
        });
    }
  }

  if(isLiDAR){
#if USE_LIVOX_CUSTOM_MSG
    subLivoxCustom_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
      "/okvis/livox", rclcpp::SensorDataQoS(),
      std::bind(&Subscriber::livoxCustomCallback, this, std::placeholders::_1));
    RCLCPP_INFO(node_->get_logger(), "Subscribed to Livox custom message");
#else
    subLiDAR_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/okvis/lidar", rclcpp::SensorDataQoS()
      std::bind(&Subscriber::lidarCallback, this, std::placeholders::_1));
#endif
  }
}

void Subscriber::shutdown() {
  // stop callbacks
  for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
#if USE_COMPRESSED_IMAGE
    if (compressedImageSubscribers_[i]) {
      compressedImageSubscribers_[i].reset();
    }
#else
    imageSubscribers_[i].shutdown();
#endif
  }
  subImu_.reset();
}

void Subscriber::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg,
                               unsigned int cameraIndex, bool isColour)
{
  static bool logged = false;
  if (!logged) {
    RCLCPP_INFO(node_->get_logger(),
                "imageCallback cam=%u stamp=%u.%u", cameraIndex,
                msg->header.stamp.sec, msg->header.stamp.nanosec);
    logged = true;
  }
  cv_bridge::CvImageConstPtr cv_ptr;  
  cv::Mat raw;
  try
  {
    if(!isColour){
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
    } else {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    }
    raw = cv_ptr->image;
  }
  catch (cv_bridge::Exception& e)
  {
    RCLCPP_ERROR(node_->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat filtered;
  filtered = raw.clone();

  // adapt timestamp
  okvis::Time t(msg->header.stamp.sec, msg->header.stamp.nanosec);

  // insert
  std::lock_guard<std::mutex> lock(time_mutex_);
  imagesReceived_.at(cameraIndex)[t.toNSec()] = filtered;
  
  // try sync
  synchronizeData();
}

void Subscriber::compressedImageCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg,
                                         unsigned int cameraIndex, bool isColour)
{
  cv::Mat raw;
  try
  {
    // Decode compressed image
    cv::Mat decoded = cv::imdecode(cv::Mat(msg->data), isColour ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
    if (decoded.empty()) {
      RCLCPP_ERROR(node_->get_logger(),
                   "Failed to decode compressed image cam=%u stamp=%u.%u format='%s' bytes=%zu",
                   cameraIndex, msg->header.stamp.sec, msg->header.stamp.nanosec,
                   msg->format.c_str(), msg->data.size());
      return;
    }
    
    // Convert to proper format if needed
    if (isColour && decoded.channels() == 3) {
      cv::cvtColor(decoded, raw, cv::COLOR_BGR2RGB);
    } else {
      raw = decoded;
    }
  }
  catch (cv::Exception& e)
  {
    RCLCPP_ERROR(node_->get_logger(), "cv exception: %s", e.what());
    return;
  }
  cv::Mat filtered;
  filtered = raw.clone();

  // adapt timestamp
  okvis::Time t(msg->header.stamp.sec, msg->header.stamp.nanosec);

  // insert
  std::lock_guard<std::mutex> lock(time_mutex_);
  imagesReceived_.at(cameraIndex)[t.toNSec()] = filtered;
  
  // try sync
  synchronizeData();
}

void Subscriber::imuCallback(const sensor_msgs::msg::Imu& msg)
{
  static bool logged = false;
  if (!logged) {
    RCLCPP_INFO(node_->get_logger(),
                "imuCallback stamp=%u.%u", msg.header.stamp.sec,
                msg.header.stamp.nanosec);
    logged = true;
  }
  // construct measurement
  okvis::Time timestamp(msg.header.stamp.sec, msg.header.stamp.nanosec);
  timestamp -= okvis::Duration(0.045);
  Eigen::Vector3d acc(msg.linear_acceleration.x, msg.linear_acceleration.y,
                      msg.linear_acceleration.z);
  Eigen::Vector3d gyr(msg.angular_velocity.x, msg.angular_velocity.y,
                      msg.angular_velocity.z);                    
  
  // forward to estimator
  viInterface_->addImuMeasurement(timestamp, acc, gyr);
  
   // also forward for realtime prediction
  if(seInterface_) {
    seInterface_->realtimePredict(timestamp, acc, gyr);
  }
  else if(publisher_) {
    publisher_->realtimePredictAndPublish(timestamp, acc, gyr);
  }
}

void Subscriber::depthCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg, 
                               unsigned int cameraIndex){
  static bool logged = false;
  if (!logged) {
    RCLCPP_INFO(node_->get_logger(),
                "depthCallback cam=%u stamp=%u.%u", cameraIndex,
                msg->header.stamp.sec, msg->header.stamp.nanosec);
    logged = true;
  }
  // TODO now we work only with one camera of depth images,
  // as it is what the submapping interface accepts
  cv_bridge::CvImageConstPtr cv_ptr;
  cv::Mat raw;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
    raw = cv_ptr->image;
  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(node_->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  okvis::Time t(msg->header.stamp.sec, msg->header.stamp.nanosec);

  if(!viInterface_->addDepthMeasurement(t, raw)){
    //LOG(WARNING) << "Dropped last depth image frame for okvis interface";
  }

  // try sync
  std::lock_guard<std::mutex> lock(time_mutex_);
  depthImagesReceived_.at(cameraIndex)[t.toNSec()] = raw;
  //add here the depth images to the receiver
  synchronizeData();
}

void Subscriber::compressedDepthCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg, 
                                         unsigned int cameraIndex){
  static bool logged = false;
  if (!logged) {
    RCLCPP_INFO(node_->get_logger(),
                "compressedDepthCallback cam=%u stamp=%u.%u", cameraIndex,
                msg->header.stamp.sec, msg->header.stamp.nanosec);
    logged = true;
  }
  cv::Mat raw;
  try {
    // Decode compressed depth image (typically 16-bit or 32-bit)
    cv::Mat decoded = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_UNCHANGED);
    if (decoded.empty()) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to decode compressed depth image");
      return;
    }
    
    // Convert to 32FC1 if needed
    if (decoded.type() == CV_16UC1) {
      decoded.convertTo(raw, CV_32FC1, 1.0/1000.0); // Convert mm to meters
    } else if (decoded.type() == CV_32FC1) {
      raw = decoded;
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Unexpected depth image type: %d", decoded.type());
      return;
    }
  } catch (cv::Exception& e) {
    RCLCPP_ERROR(node_->get_logger(), "cv exception: %s", e.what());
    return;
  }

  okvis::Time t(msg->header.stamp.sec, msg->header.stamp.nanosec);

  if(!viInterface_->addDepthMeasurement(t, raw)){
    //LOG(WARNING) << "Dropped last depth image frame for okvis interface";
  }

  // try sync
  std::lock_guard<std::mutex> lock(time_mutex_);
  depthImagesReceived_.at(cameraIndex)[t.toNSec()] = raw;
  //add here the depth images to the receiver
  synchronizeData();
}

#if USE_LIVOX_CUSTOM_MSG
void Subscriber::livoxCustomCallback(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg){
  Eigen::Vector3d ray;
  
  // Use header timestamp for all points in this message
  okvis::Time t(msg->header.stamp.sec, msg->header.stamp.nanosec);

  for(const auto& point : msg->points) {
    ray << point.x, point.y, point.z;
    
    if(seInterface_) {
      if(!seInterface_->addLidarMeasurement(t, ray)) {
        LOG(WARNING) << "Dropped last lidar measurement from SubmappingInterface.";
      }
    }
    viInterface_->addLidarMeasurement(t, ray);
  }
}
#endif

void Subscriber::lidarCallback(const sensor_msgs::msg::PointCloud2& msg){
  std::cout << "lidarCallback stamp=" << msg.header.stamp.sec << "." << msg.header.stamp.nanosec << std::endl;
  static bool logged = false;
  if (!logged) {
    RCLCPP_INFO(node_->get_logger(),
                "lidarCallback stamp=%u.%u", msg.header.stamp.sec,
                msg.header.stamp.nanosec);
    logged = true;
  }

  // Check which type of LiDAR Point Cloud it is
  bool is_blk = (okvis::pointcloud_ros::has_field(msg, "stamp_high") && okvis::pointcloud_ros::has_field(msg, "stamp_low"));
  bool is_hesai = okvis::pointcloud_ros::has_field(msg, "timestamp");

  if(!(is_blk || is_hesai)){ // Default; use header timestamp
    Eigen::Vector3d ray;
    okvis::Time t(msg.header.stamp.sec, msg.header.stamp.nanosec);

    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(msg, pcl_pc2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
    pcl::PointCloud<pcl::PointXYZ>::iterator it = temp_cloud->begin();

    for (/*it*/; it != temp_cloud->end(); it++){
      ray << it->x, it->y, it->z;
      if(!seInterface_->addLidarMeasurement(t, ray)) LOG(WARNING) << "Dropped last lidar measurement from SubmappingInterface.";
      viInterface_->addLidarMeasurement(t, ray);
    }
  }
  else{
    std::vector<okvis::Time, Eigen::aligned_allocator<okvis::Time>> timestamps;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> rays;

    if(is_hesai){
      okvis::pointcloud_ros::hesai_lidar2points(msg, timestamps, rays);
    }
    else if(is_blk){
      okvis::pointcloud_ros::blk_lidar2points(msg, timestamps, rays);
    }

    for(size_t i = 0; i < timestamps.size(); i++){
      if(!seInterface_->addLidarMeasurement(timestamps[i], rays[i])) LOG(WARNING) << "Dropped last lidar measurement from SubmappingInterface.";
      viInterface_->addLidarMeasurement(timestamps[i], rays[i]);
    }
  }
}

void Subscriber::synchronizeData() {
  const double syncToleranceSec =
      parameters_.camera.timestamp_tolerance > 0.0
          ? parameters_.camera.timestamp_tolerance
          : kDefaultSyncToleranceSec;
  const uint64_t syncToleranceNs = static_cast<uint64_t>(syncToleranceSec * 1e9);

  const int numCameras = imagesReceived_.size();
  std::vector<int> slamCameras;
  slamCameras.reserve(numCameras);
  for (int i = 0; i < numCameras; ++i) {
    const bool isSlamCamera = parameters_.nCameraSystem.cameraType(i).isUsed;
    const bool isSyncCamera = parameters_.camera.sync_cameras.count(size_t(i)) > 0;
    if (isSlamCamera && isSyncCamera) {
      slamCameras.push_back(i);
    }
  }
  if (slamCameras.empty()) {
    return;
  }

  std::set<uint64_t> allTimes;
  for (const int camId : slamCameras) {
    for (const auto& entry : imagesReceived_.at(camId)) {
      allTimes.insert(entry.first);
    }
  }

  bool synchronizedPacketFound = false;
  for (const auto& time : allTimes) {
    // note: ordered old to new
    std::vector<uint64_t> syncedTimes(numCameras, 0);
    uint64_t depthSyncedTime = 0; // Assume there is only one depth image source for SI.
    std::map<uint64_t, cv::Mat> images;
    std::map<uint64_t, cv::Mat> depthImages;

    std::map<uint64_t, std::pair<okvis::Time, cv::Mat>> timestampedImages;
    std::map<uint64_t, std::pair<okvis::Time, cv::Mat>> timestampedDepthImages;

    okvis::Time tcheck;
    tcheck.fromNSec(time);
    bool synced = true;
    for (const int camId : slamCameras) {
      bool syncedi = false;
      for (const auto& entry : imagesReceived_.at(camId)) {
        const uint64_t dtNs = time >= entry.first ? (time - entry.first) : (entry.first - time);
        if (dtNs < syncToleranceNs && time >= entry.first) {
          okvis::Time ti;
          ti.fromNSec(entry.first);
          syncedTimes.at(camId) = entry.first;
          images[camId] = imagesReceived_.at(camId).at(entry.first);
          timestampedImages[camId] = std::make_pair(ti, imagesReceived_.at(camId).at(entry.first));
          syncedi = true;
          break;
        }
      }

      if (!syncedi) {
        synced = false;
        break;
      }
    }

    bool syncedDepth = true;
    if (syncDepthImages_) {
      syncedDepth = false;
      for (int i = 0; i < numCameras; ++i) {
        for (const auto& entry : depthImagesReceived_.at(i)) {
          const uint64_t dtNs = time >= entry.first ? (time - entry.first) : (entry.first - time);
          // There is a higher unsynchronization between depth images and IR images from the realsense.
          if (dtNs < syncToleranceNs && time >= entry.first) {
            okvis::Time tdepth;
            tdepth.fromNSec(entry.first);
            depthSyncedTime = entry.first;
            depthImages[i] = depthImagesReceived_.at(i).at(entry.first);
            timestampedDepthImages[i] = std::make_pair(tdepth, depthImagesReceived_.at(i).at(entry.first));
            syncedDepth = true;
            break;
          }
        }

        if (syncedDepth) {
          // For now we assume there is only one depth image per synchronization process.
          break;
        }
      }
    }

    if (synced && syncedDepth) {
      synchronizedPacketFound = true;
      bool isProcessor = false;
      #ifdef OKVIS_USE_NN
      okvis::Processor* casted_processor = dynamic_cast<okvis::Processor*>(viInterface_);
      if (casted_processor) {
        isProcessor = true;
        if(!casted_processor->addImages(timestampedImages, timestampedDepthImages)) {
          LOG(WARNING) << "Frame not added to Processor at t="<< timestampedImages.at(0).first;
        }
      }
      else if(!viInterface_->addImages(tcheck, images, depthImages)) {
        LOG(WARNING) << "Frame not added at t="<< tcheck;
      }
      #else
      if(!viInterface_->addImages(tcheck, images, depthImages)) {
        LOG(WARNING) << "Frame not added at t="<< tcheck;
      }
      #endif
      if(!isProcessor && syncDepthImages_) {
        // If we have depth images, then we send the data to SI from here.
        std::map<size_t, std::vector<okvis::CameraMeasurement>> cameraMeasurements;

        for(const auto& it : timestampedImages) {
          okvis::CameraMeasurement cameraMeasure;
          cameraMeasure.timeStamp = tcheck;
          cameraMeasure.measurement.image = it.second.second.clone();
          cameraMeasure.sensorId = it.first;
          cameraMeasurements[it.first] = {cameraMeasure};
        }

        for(const auto& it : timestampedDepthImages) {
          if(cameraMeasurements.find(it.first) != cameraMeasurements.end()) {
            bool imageAdded = false;
            for(auto& image : cameraMeasurements.at(it.first)) {
              if(image.timeStamp == it.second.first){
                image.measurement.depthImage = it.second.second.clone();
                imageAdded = true;
              }
            }

            if(!imageAdded) {
              okvis::CameraMeasurement cameraMeasure;
              cameraMeasure.timeStamp = it.second.first;
              cameraMeasure.measurement.depthImage = it.second.second.clone();
              cameraMeasure.sensorId = it.first;
              cameraMeasurements[it.first] = {cameraMeasure};
            }
          } else {
            okvis::CameraMeasurement cameraMeasure;
            cameraMeasure.timeStamp = it.second.first;
            cameraMeasure.measurement.depthImage = it.second.second.clone();
            cameraMeasure.sensorId = it.first;
            cameraMeasurements[it.first] = {cameraMeasure};
          }
        }

        if(!seInterface_->addDepthMeasurement(cameraMeasurements)) {
          LOG(WARNING) << "Frame not added to SI at t="<< cameraMeasurements.at(0)[0].timeStamp;
        }
      }

      // Remove all older buffered data up to the synchronized sample.
      for (const int camId : slamCameras) {
        auto end = imagesReceived_.at(camId).find(syncedTimes.at(camId));
        if (end != imagesReceived_.at(camId).end()) {
          ++end;
        }
        imagesReceived_.at(camId).erase(imagesReceived_.at(camId).begin(), end);
      }

      if(syncDepthImages_) {
        for(int i=0; i < numCameras; ++i) {
          auto end = std::find_if(depthImagesReceived_.at(i).begin(), depthImagesReceived_.at(i).end(),
                                 [depthSyncedTime](const auto& x) { return x.first > depthSyncedTime; });
          depthImagesReceived_.at(i).erase(depthImagesReceived_.at(i).begin(), end);
        }
      }
    }
  }

  static size_t unsyncedCycles = 0;
  if (!synchronizedPacketFound && !allTimes.empty()) {
    ++unsyncedCycles;
    if (unsyncedCycles % 200 == 0) {
      std::ostringstream bufferInfo;
      for (const int camId : slamCameras) {
        if (!imagesReceived_.at(camId).empty()) {
          const uint64_t firstTs = imagesReceived_.at(camId).begin()->first;
          const uint64_t lastTs = imagesReceived_.at(camId).rbegin()->first;
          bufferInfo << " cam" << camId << "[" << imagesReceived_.at(camId).size()
                     << "] first=" << firstTs << " last=" << lastTs;
        } else {
          bufferInfo << " cam" << camId << "[0]";
        }
      }
      LOG(WARNING) << "No synchronized camera packet found yet. tol="
                   << syncToleranceSec << "s." << bufferInfo.str();
    }
  } else if (synchronizedPacketFound) {
    unsyncedCycles = 0;
  }
}

} // namespace okvis