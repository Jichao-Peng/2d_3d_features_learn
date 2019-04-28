//
// Created by leo on 19-4-23.
//

#include "feature3d.h"

namespace Feature3d
{
    void Feature3d::Run(const string &filePath, Method method)
    {
        cout << "Reading file: " << filePath << endl;
        PointCloud<PointXYZRGB>::Ptr pCloud(new PointCloud<PointXYZRGB>);
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(filePath, *pCloud) == -1) // load the file
        {
            PCL_ERROR ("Couldn't read file");
            return;
        }
        std::cout << "Points: " << pCloud->points.size() << std::endl;

        //SIFT特征子提取 TODO:可能是因为参数的原因，无法提取出有效的关键点
        if (method == SIFT)
        {
            PointCloud<PointWithScale>::Ptr pSift(new PointCloud<PointWithScale>());
            DetectSIFTKeypoints(pCloud,pSift);
            std::cout << "SIFT KeyPoints: " << pSift->points.size() << std::endl;

            //查看点云
            pcl::visualization::PCLVisualizer viewer("PCL Viewer");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> cloudColorHandler(pCloud, 255, 0, 0);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithScale> keypointsColorHandler(pSift, 0, 255, 0);
            viewer.setBackgroundColor(0.0, 0.0, 0.0);
            viewer.addPointCloud(pSift, keypointsColorHandler, "keypoints");
            viewer.addPointCloud(pCloud, cloudColorHandler, "cloud");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "keypoints");

            while (!viewer.wasStopped())
            {
                viewer.spinOnce();
            }
        }
        else if (method == Harris)
        {
            //Harris特征子提取
            PointCloud<PointXYZI>::Ptr pHarris(new PointCloud<PointXYZI>());//这里只能使用PointXYZI的点云类型
            DetectHarrisKeypoints(pCloud, pHarris);//这个阈值调整好像没什么用
            std::cout << "Harris KeyPoints: " << pHarris->points.size() << std::endl;

            //查看点云
            pcl::visualization::PCLVisualizer viewer("PCL Viewer");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> cloudColorHandler(pCloud, 255, 0, 0);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> keypointsColorHandler(pHarris, 0, 255,0);
            viewer.setBackgroundColor(0.0, 0.0, 0.0);
            viewer.addPointCloud(pHarris, keypointsColorHandler, "keypoints");
            viewer.addPointCloud(pCloud, cloudColorHandler, "cloud");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "keypoints");

            while (!viewer.wasStopped())
            {
                viewer.spinOnce();
            }
        }
        //NARF
        else if(method == NARF)
        {
            PointCloud<PointXYZ>::Ptr pNarf(new PointCloud<PointXYZ>());//这里只能使用PointXYZI的点云类型
            PointCloud<PointXYZ>::Ptr pCloudXYZ(new PointCloud<PointXYZ>());
            copyPointCloud(*pCloud,*pCloudXYZ);
            DetectNARFKeypoints(pCloudXYZ, pNarf);
            std::cout << "ISS KeyPoints: " << pNarf->points.size() << std::endl;
            //查看点云
            pcl::visualization::PCLVisualizer viewer("PCL Viewer");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> cloudColorHandler(pCloud, 255, 0, 0);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypointsColorHandler(pNarf, 0, 255, 0);
            viewer.setBackgroundColor(0.0, 0.0, 0.0);
            viewer.addPointCloud(pNarf, keypointsColorHandler, "keypoints");
            viewer.addPointCloud(pCloud, cloudColorHandler, "cloud");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "keypoints");

            while (!viewer.wasStopped())
            {
                viewer.spinOnce();
            }
        }
        //ISS
        else if(method == ISS)
        {
            PointCloud<PointXYZRGB>::Ptr pIss(new PointCloud<PointXYZRGB>());//这里只能使用PointXYZI的点云类型
            DetectISSKeypoints(pCloud, pIss);
            std::cout << "ISS KeyPoints: " << pIss->points.size() << std::endl;
            //查看点云
            pcl::visualization::PCLVisualizer viewer("PCL Viewer");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> cloudColorHandler(pCloud, 255, 0, 0);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> keypointsColorHandler(pIss, 0, 255,
                                                                                                   0);
            viewer.setBackgroundColor(0.0, 0.0, 0.0);
            viewer.addPointCloud(pIss, keypointsColorHandler, "keypoints");
            viewer.addPointCloud(pCloud, cloudColorHandler, "cloud");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "keypoints");

            while (!viewer.wasStopped())
            {
                viewer.spinOnce();
            }
        }
    }

    void Feature3d::DetectSIFTKeypoints(PointCloud<PointXYZRGB>::Ptr &cloud, PointCloud<PointWithScale>::Ptr &keypoints)
    {

        // Estimate the normals
        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointNormal> normalEstimation;
        pcl::PointCloud<pcl::PointNormal>::Ptr normalCloud (new pcl::PointCloud<pcl::PointNormal>);
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr normalKdTree(new pcl::search::KdTree<pcl::PointXYZRGB>());

        normalEstimation.setInputCloud(cloud);
        normalEstimation.setSearchMethod(normalKdTree);
        normalEstimation.setRadiusSearch(0.02);
        normalEstimation.compute(*normalCloud);

        for(size_t i = 0; i<normalCloud->points.size(); ++i)
        {
            normalCloud->points[i].x = cloud->points[i].x;
            normalCloud->points[i].y = cloud->points[i].y;
            normalCloud->points[i].z = cloud->points[i].z;
        }

        float minScale = 0.001f;
        int nrOctaves = 3;
        int nrScalesPerOctave = 4;
        float minContrast = 0.0001f;

        SIFTKeypoint<PointNormal, PointWithScale> siftDetect;
        search::KdTree<PointNormal>::Ptr kdTree(new search::KdTree<PointNormal>());
        siftDetect.setSearchMethod(kdTree);
        siftDetect.setScales(minScale, nrOctaves, nrScalesPerOctave);
        siftDetect.setMinimumContrast(minContrast);
        siftDetect.setInputCloud(normalCloud);
        siftDetect.compute(*keypoints);
    }


    void Feature3d::DetectHarrisKeypoints(PointCloud<PointXYZRGB>::Ptr &cloud, PointCloud<PointXYZI>::Ptr &keypoints)
    {
        float threshold = 0.0000001f;
        HarrisKeypoint3D<PointXYZRGB, PointXYZI> harrisDetect;
        harrisDetect.setNonMaxSupression(true);
        harrisDetect.setInputCloud(cloud);
        harrisDetect.setThreshold(threshold);
        harrisDetect.compute(*keypoints);
    }

    void Feature3d::DetectISSKeypoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr &keypoints)
    {
        double modelResolution = 0.001f;
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZRGB>());
        ISSKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZRGB> issDetector;
        issDetector.setSearchMethod (kdTree);
        issDetector.setSalientRadius (6 * modelResolution);
        issDetector.setNonMaxRadius (4 * modelResolution);
        issDetector.setThreshold21 (0.9);
        issDetector.setThreshold32 (0.9);
        issDetector.setMinNeighbors (5);
        issDetector.setNumberOfThreads (4);
        issDetector.setInputCloud (cloud);
        issDetector.compute (*keypoints);
    }

    void Feature3d::DetectNARFKeypoints(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                        pcl::PointCloud<pcl::PointXYZ>::Ptr &keypoints)
    {
        shared_ptr<RangeImage> rangeImage(new RangeImage);
        pcl::PointCloud<PointXYZ>& pointCloud = *cloud;

        Eigen::Affine3f sceneSensorPose = Eigen::Affine3f(Eigen::Translation3f(pointCloud.sensor_origin_[0], pointCloud.sensor_origin_[1], pointCloud.sensor_origin_[2]))*Eigen::Affine3f (pointCloud.sensor_orientation_);
        rangeImage->createFromPointCloud (pointCloud, pcl::deg2rad (0.5f), pcl::deg2rad (360.0f), pcl::deg2rad (180.0f), sceneSensorPose, pcl::RangeImage::CAMERA_FRAME, 0.0, 0.0f, 1);
        rangeImage->setUnseenToMaxRange();

        pcl::visualization::RangeImageVisualizer rangeImageViewer ("Range image");
        rangeImageViewer.showRangeImage (*rangeImage);

        while (!rangeImageViewer.wasStopped())
        {
            rangeImageViewer.spinOnce();
        }

        pcl::RangeImageBorderExtractor rangeImageBorderExtractor;
        pcl::NarfKeypoint narfKeypointDetector;
        narfKeypointDetector.setRangeImageBorderExtractor (&rangeImageBorderExtractor);
        narfKeypointDetector.setRangeImage (rangeImage.get());
        narfKeypointDetector.getParameters().support_size = 0.02f;
        narfKeypointDetector.setRadiusSearch(0.001);

        pcl::PointCloud<int> keypointIndices;
        narfKeypointDetector.compute (keypointIndices);

        keypoints->points.resize(keypointIndices.points.size());
        for (size_t i=0; i<keypointIndices.points.size(); ++i)
        {
            keypoints->points[i].getVector3fMap () = rangeImage->points[keypointIndices.points[i]].getVector3fMap();
        }
    }
}






