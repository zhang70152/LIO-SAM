#include "utility.h"
#include "lio_sam/cloud_info.h"

#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <pcl/registration/ndt.h>


/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                    (float, x, x) (float, y, y)
                                    (float, z, z) (float, intensity, intensity)
                                    (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                    (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class globalLocalizor : public ParamServer
{

public:

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;

    ros::Subscriber subLaserCloudInfo;

    lio_sam::cloud_info cloudInfo;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;



    float transformTobeMapped[6];

    std::mutex mtx;

    double timeLastProcessing = -1;



    pcl::PointCloud<PointType>::Ptr cloudGlobalMap;
    pcl::PointCloud<PointType>::Ptr cloudGlobalMapDS;
    pcl::PointCloud<PointType>::Ptr cloudScanForInitialize;

    ros::Subscriber subIniPoseFromRviz;
    ros::Subscriber subLaserOdometry;
    ros::Publisher pubLaserCloudInWorld;
    ros::Publisher pubMapWorld;


    float transformInTheWorld[6];// the pose in the world, i.e. the prebuilt map
    float tranformOdomToWorld[6];


    tf::TransformBroadcaster tfOdom2Map;
    std::mutex mtxtranformOdomToWorld;
    std::mutex mtx_general;

    tf::TransformListener tfListener;

    std::deque<nav_msgs::Odometry> odom_queue_;
    std::deque<lio_sam::cloud_info> cloud_queue_;


    geometry_msgs::PoseStamped poseOdomToMap;
    ros::Publisher pubOdomToMapPose;
    tf::Transform map_to_odom_;
    tf::Transform frozen_map_to_baselink;

    tf::StampedTransform lidar2Baselink;

    double last_process_time_;
    globalLocalizor()
    {
        subLaserCloudInfo = nh.subscribe<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1, &globalLocalizor::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

        subIniPoseFromRviz = nh.subscribe("/initialpose", 8, &globalLocalizor::initialPoseCallback, this);
        subLaserOdometry   = nh.subscribe("lio_sam/mapping/odometry", 2, &globalLocalizor::laserOdomCallBack, this);

        pubOdomToMapPose = nh.advertise<geometry_msgs::PoseStamped>("lio_sam/global_localization/pose_odomTo_map", 1);
        // pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
        pubMapWorld = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/global_localization/frozen_map", 1);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

        allocateMemory();

        tf::Transform initial_map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
        map_to_odom_ = initial_map_to_odom;

        if(lidarFrame != baselinkFrame)
        {
            try
            {
                tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
                tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink);
            }
            catch (tf::TransformException ex)
            {
                ROS_ERROR("%s",ex.what());
            }
        }
        last_process_time_ = 0;
    }

    void allocateMemory()
    {
        cloudGlobalMap.reset(new pcl::PointCloud<PointType>());//addded by gc
	    cloudGlobalMapDS.reset(new pcl::PointCloud<PointType>());//added
        cloudScanForInitialize.reset(new pcl::PointCloud<PointType>());
        resetLIO();
 
        for (int i = 0; i < 6; ++i){
            transformInTheWorld[i] = 0;
        }

        for (int i = 0; i < 6; ++i){
            tranformOdomToWorld[i] = 0;
        }
    
        loadGlobalMap();

    }
    
    void resetLIO()
    {

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

    }

    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        cloud_queue_.push_back(*msgIn);
        if(last_process_time_ == 0 ) {
            last_process_time_ = msgIn->header.stamp.toSec();
            return;
        }

        if(msgIn->header.stamp.toSec() - last_process_time_ > 3.0) {
            process();
            last_process_time_ = msgIn->header.stamp.toSec();
        }

        pubMapToOdom(msgIn->header.stamp);
        

    }


    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

        for (int i = 0; i < cloudSize; ++i){

            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }


    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }




    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);


        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
    }



    void loadGlobalMap()
    {
        pcl::io::loadPCDFile(std::getenv("HOME") + savePCDDirectory + "GlobalMap.pcd", *cloudGlobalMap);

        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(cloudGlobalMap);
        downSizeFilterICP.filter(*cloud_temp);

        *cloudGlobalMapDS = *cloud_temp;

        std::cout << "test 0.01  the size of global cloud: " << cloudGlobalMap->points.size() << std::endl;
        std::cout << "test 0.02  the size of global map after filter: " << cloudGlobalMapDS->points.size() << std::endl;
    }



    void ICPMatch(pcl::PointCloud<PointType>::Ptr cloud_for_match, nav_msgs::Odometry synced_odom)
    {
        if(cloud_for_match->points.size() == 0) {
            std::cout<<"No thing to match, skip."<<std::endl;
            return;
        }
    

        // update latest frozen map pose for initial guess
        {
            tf::StampedTransform frozenMap2Lidar;
            try
            {
                tfListener.waitForTransform(mapFrame, "center_3D_lidar", synced_odom.header.stamp, ros::Duration(0.1));
                tfListener.lookupTransform(mapFrame, "center_3D_lidar", synced_odom.header.stamp, frozenMap2Lidar);
            } 
            catch (tf::TransformException ex)
            {
                ROS_ERROR("%s",ex.what());
            }
            double roll1, pitch1, yaw1;
            tf::Matrix3x3(frozenMap2Lidar.getRotation()).getRPY(roll1, pitch1, yaw1);
            transformInTheWorld[0] = roll1;
            transformInTheWorld[1] = pitch1;
            transformInTheWorld[2] = yaw1;
            transformInTheWorld[3] = frozenMap2Lidar.getOrigin().getX();
            transformInTheWorld[4] = frozenMap2Lidar.getOrigin().getY();
            transformInTheWorld[5] = frozenMap2Lidar.getOrigin().getZ(); 
        }

        PointTypePose thisPose6DInWorld = trans2PointTypePose(transformInTheWorld);
        Eigen::Affine3f T_thisPose6DInWorld = pclPointToAffine3f(thisPose6DInWorld);
    

   
        // First do ndt.
        pcl::NormalDistributionsTransform<PointType, PointType> ndt;
        ndt.setTransformationEpsilon(0.01);
        ndt.setResolution(1.0);
        ndt.setInputSource(cloud_for_match);
        ndt.setInputTarget(cloudGlobalMapDS);
        pcl::PointCloud<PointType>::Ptr unused_result_0(new pcl::PointCloud<PointType>());
        ndt.align(*unused_result_0, T_thisPose6DInWorld.matrix());


        //use the outcome of ndt as the initial guess for ICP
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(100);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);
        icp.setInputSource(cloud_for_match);
        icp.setInputTarget(cloudGlobalMapDS);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result, ndt.getFinalTransformation());


        //update pose in odom frame.
        transformTobeMapped[3] = synced_odom.pose.pose.position.x;
        transformTobeMapped[4] = synced_odom.pose.pose.position.y;
        transformTobeMapped[5] = synced_odom.pose.pose.position.z;

        tf::Quaternion q(
            synced_odom.pose.pose.orientation.x,
            synced_odom.pose.pose.orientation.y,
            synced_odom.pose.pose.orientation.z,
            synced_odom.pose.pose.orientation.w);
        tf::Matrix3x3 m(q);
        double roll2, pitch2, yaw2;
        m.getRPY(roll2, pitch2, yaw2);

        transformTobeMapped[0] = roll2;
        transformTobeMapped[1] = pitch2;
        transformTobeMapped[2] = yaw2;
        PointTypePose thisPose6DInOdom = trans2PointTypePose(transformTobeMapped);
	    
        Eigen::Affine3f T_thisPose6DInOdom = pclPointToAffine3f(thisPose6DInOdom);  







        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
        {
            std::cout << "---------Initializing Fail, score:" <<icp.getFitnessScore() << std::endl;
            return;
        } else{
            std::cout << "-------Initializing Succeed score:" <<icp.getFitnessScore() <<std::endl;

            Eigen::Affine3f T_thisPose6DInMap;
            T_thisPose6DInMap = icp.getFinalTransformation();
            float x_g, y_g, z_g, R_g, P_g, Y_g;
            pcl::getTranslationAndEulerAngles (T_thisPose6DInMap, x_g, y_g, z_g, R_g, P_g, Y_g);
            transformInTheWorld[0] = R_g;
            transformInTheWorld[1] = P_g;
            transformInTheWorld[2] = Y_g;
            transformInTheWorld[3] = x_g;
            transformInTheWorld[4] = y_g;
            transformInTheWorld[5] = z_g;


            Eigen::Affine3f transOdomToMap = T_thisPose6DInMap * T_thisPose6DInOdom.inverse();
            float deltax, deltay, deltaz, deltaR, deltaP, deltaY;
            pcl::getTranslationAndEulerAngles (transOdomToMap, deltax, deltay, deltaz, deltaR, deltaP, deltaY);

            mtxtranformOdomToWorld.lock();
                //renew tranformOdomToWorld
            tranformOdomToWorld[0] = deltaR;
            tranformOdomToWorld[1] = deltaP;
            tranformOdomToWorld[2] = deltaY;
            tranformOdomToWorld[3] = deltax;
            tranformOdomToWorld[4] = deltay;
            tranformOdomToWorld[5] = deltaz;
            mtxtranformOdomToWorld.unlock();

            publishCloud(&pubLaserCloudInWorld, unused_result, synced_odom.header.stamp, mapFrame);


            // static tf
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (T_thisPose6DInMap, x, y, z, roll, pitch, yaw);
            tf::Transform frozen_map_to_base_link = tf::Transform(tf::createQuaternionFromRPY(roll, pitch, yaw), tf::Vector3(x, y, z));



            pcl::getTranslationAndEulerAngles (T_thisPose6DInOdom, x, y, z, roll, pitch, yaw);
            tf::Transform map2Baselink = tf::Transform(tf::createQuaternionFromRPY(roll, pitch, yaw), tf::Vector3(x, y, z));


            map_to_odom_ = frozen_map_to_base_link * map2Baselink.inverse();
            frozen_map_to_baselink = frozen_map_to_base_link;

            double roll1, pitch1, yaw1;

            tf::Matrix3x3(map_to_odom_.getRotation()).getRPY(roll1, pitch1, yaw1);
            std::cout<<"match result x:"<<map_to_odom_.getOrigin().getX()
                      <<"    y:"<<map_to_odom_.getOrigin().getY()
                      <<"    z:"<<map_to_odom_.getOrigin().getZ()
                      <<"    roll:"<<roll1
                      <<"    pitch:"<<pitch1
                      <<"    yaw:"<<yaw1<<std::endl;
        }

    }





    void laserOdomCallBack(const nav_msgs::OdometryConstPtr& odom_msg)
    {
        odom_queue_.push_back(*odom_msg);
    }



    void initialPoseCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg)
    {
        tf::Transform map_to_base_link;
        tf::Transform map_to_lidar;
        tf::poseMsgToTF(pose_msg->pose.pose, map_to_base_link);
        map_to_lidar = map_to_base_link * lidar2Baselink.inverse();
        float x = map_to_lidar.getOrigin().getX();
        float y = map_to_lidar.getOrigin().getY();
        float z = map_to_lidar.getOrigin().getZ();
        tf::Quaternion q_global = map_to_lidar.getRotation();
        double roll_global; double pitch_global; double yaw_global;

        tf::Matrix3x3(q_global).getRPY(roll_global, pitch_global, yaw_global);
        //global transformation
        transformInTheWorld[0] = roll_global;
        transformInTheWorld[1] = pitch_global;
        transformInTheWorld[2] = yaw_global;
        transformInTheWorld[3] = x;
        transformInTheWorld[4] = y;
        transformInTheWorld[5] = z;
        PointTypePose thisPose6DInWorld = trans2PointTypePose(transformInTheWorld);
        Eigen::Affine3f T_thisPose6DInWorld = pclPointToAffine3f(thisPose6DInWorld);
        //Odom transformation
        PointTypePose thisPose6DInOdom = trans2PointTypePose(transformTobeMapped);
        Eigen::Affine3f T_thisPose6DInOdom = pclPointToAffine3f(thisPose6DInOdom);
        //transformation: Odom to Map
        Eigen::Affine3f T_OdomToMap = T_thisPose6DInWorld * T_thisPose6DInOdom.inverse();
        float delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw;
        pcl::getTranslationAndEulerAngles (T_OdomToMap, delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw);

        mtxtranformOdomToWorld.lock();
        //keep for co-operate the initializing and lio, not useful for the present
        tranformOdomToWorld[0] = delta_roll;
        tranformOdomToWorld[1] = delta_pitch;
        tranformOdomToWorld[2] = delta_yaw;
        tranformOdomToWorld[3] = delta_x;
        tranformOdomToWorld[4] = delta_y;
        tranformOdomToWorld[5] = delta_z;
        mtxtranformOdomToWorld.unlock();

        // static tf
        //Eigen::Affine3f transMap2Odom = T_OdomToMap.inverse();
        Eigen::Affine3f transMap2Odom = T_OdomToMap;
        float x1, y1, z1, roll1, pitch1, yaw1;
        pcl::getTranslationAndEulerAngles (transMap2Odom, x1, y1, z1, roll1, pitch1, yaw1);
        tf::Transform corrected_map_to_odom = tf::Transform(tf::createQuaternionFromRPY(roll1, pitch1, yaw1), tf::Vector3(x1, y1, z1));
        map_to_odom_ = corrected_map_to_odom;

        process();

    }

    void globalLocalizeThread()
    {
        ros::Rate rate(3);
        while (ros::ok())
        {
            std::cout << "Run once " << std::endl;//do nothing, wait for a new initial guess
            process();
            rate.sleep();
        }
    }

    void pubMapToOdom(ros::Time time_stamp)
    {
        ros::Time pub_timestamp;
        tf::StampedTransform odom2Baselink;
        try
        {
            tfListener.waitForTransform(odometryFrame, baselinkFrame, ros::Time::now() - ros::Duration(0.1), ros::Duration(0.15));
            tfListener.lookupTransform(odometryFrame, baselinkFrame, ros::Time::now() - ros::Duration(0.1), odom2Baselink);
            pub_timestamp = odom2Baselink.stamp_;
        } 
        catch (tf::TransformException ex)
        {
            ROS_ERROR("%s",ex.what());
            pub_timestamp = ros::Time::now();
        }

        static tf::TransformBroadcaster tfMap2Odom;
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom_, pub_timestamp, mapFrame, odometryFrame));    
    }

    void visualizeFrozenlMapThread()
    {
        ros::Rate rate(0.5);
        while (ros::ok()){
            rate.sleep();
            publishCloud(&pubMapWorld, cloudGlobalMap, ros::Time::now(), mapFrame);
        }
    }


    void process()
    {
        while(odom_queue_.size() >= 2) {
            odom_queue_.pop_front();
        }

        if(odom_queue_.empty()) {
            std::cout<<"[LOCALIZATION][WARNING] empty laser odom queue!"<<std::endl;
            return;
        }

        nav_msgs::Odometry latest_odom = odom_queue_.back();

        while(!cloud_queue_.empty())
        {
            float time_diff = cloud_queue_.front().header.stamp.toSec() - latest_odom.header.stamp.toSec();
            if( time_diff < - 0.01) {
                cloud_queue_.pop_front();
            } else {
                if(time_diff > 0.1) {
                    std::cout<<"[LOCALIZATION][WARNING] unsynchronized cloud and odom time:"<< time_diff<<std::endl;
                }
                break;
            }
        }

        if(cloud_queue_.empty()) {
            std::cout<<"[LOCALIZATION][WARNING] empty cloud odom queue!!!!"<<std::endl;
            return;
        }

        std::cout<<"queue size:"<<cloud_queue_.size()<<std::endl;
        lio_sam::cloud_info synced_cloud = cloud_queue_.front();


        pcl::fromROSMsg(synced_cloud.cloud_corner,  *laserCloudCornerLast);
        pcl::fromROSMsg(synced_cloud.cloud_surface, *laserCloudSurfLast);


        pcl::PointCloud<PointType>::Ptr cloud_for_match(new pcl::PointCloud<PointType>());
  
        downsampleCurrentScan();
    
      
        *cloud_for_match += *laserCloudCornerLastDS;
        *cloud_for_match += *laserCloudSurfLastDS;
   
        ICPMatch(cloud_for_match, latest_odom);

        laserCloudCornerLastDS->clear();
        laserCloudSurfLastDS->clear();   
        
    }


};



int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    globalLocalizor GL;

    ROS_INFO("\033[1;32m----> Global localizer Started.\033[0m");


    std::thread visualizeFrozenMapThread(&globalLocalizor::visualizeFrozenlMapThread, &GL);
    // std::thread localizeInWorldThread(&globalLocalizor::globalLocalizeThread, &GL);

    ros::spin();
    // localizeInWorldThread.join();
    visualizeFrozenMapThread.join();



    return 0;
}
