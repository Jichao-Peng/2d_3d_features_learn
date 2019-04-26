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
            DetectSIFTKeypoints(pCloud, 0.001f, 6, 10, 0.0000001f, pSift);
            std::cout << "SIFT KeyPoints: " << pSift->points.size() << std::endl;

            //查看点云
            pcl::visualization::PCLVisualizer viewer("PCL Viewer");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> cloudColorHandler(pCloud, 255, 0, 0);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithScale> keypointsColorHandler(pSift, 0,
                                                                                                        255,
                                                                                                        0);
            viewer.setBackgroundColor(0.0, 0.0, 0.0);
            viewer.addPointCloud(pSift, keypointsColorHandler, "cloud");
            viewer.addPointCloud(pCloud, cloudColorHandler, "keypoints");
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
            DetectHarrisKeypoints(pCloud, 0.000001f, pHarris);
            std::cout << "Harris KeyPoints: " << pHarris->points.size() << std::endl;

            //查看点云
            pcl::visualization::PCLVisualizer viewer("PCL Viewer");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> cloudColorHandler(pCloud, 255, 0, 0);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> keypointsColorHandler(pHarris, 0, 255,
                                                                                                   0);
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
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypointsColorHandler(pNarf, 0, 255,
                                                                                                     0);
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

    void Feature3d::DetectSIFTKeypoints(PointCloud<PointXYZRGB>::Ptr &points, float minScale, int nrOctaves,
                                        int nrScalesPerOctave, float minContrast,
                                        PointCloud<PointWithScale>::Ptr &keypoints)
    {
        SIFTKeypoint<PointXYZRGB, PointWithScale> siftDetect;
        search::KdTree<PointXYZRGB>::Ptr kdTree(new search::KdTree<PointXYZRGB>());
        siftDetect.setSearchMethod(kdTree);
        siftDetect.setScales(minScale, nrOctaves, nrScalesPerOctave);
        siftDetect.setMinimumContrast(minContrast);
        siftDetect.setInputCloud(points);
        siftDetect.compute(*keypoints);
    }


    void Feature3d::DetectHarrisKeypoints(PointCloud<PointXYZRGB>::Ptr &points, float threshold,
                                          PointCloud<PointXYZI>::Ptr &keypoints)
    {
        HarrisKeypoint3D<PointXYZRGB, PointXYZI> harrisDetect;
        harrisDetect.setNonMaxSupression(true);
        harrisDetect.setInputCloud(points);
        harrisDetect.setThreshold(threshold);
        harrisDetect.compute(*keypoints);
    }

    void Feature3d::DetectISSKeypoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points,
                                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr &keypoints)
    {
        double modelResolution = 0.001;
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZRGB>());
        ISSKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZRGB> issDetector;
        issDetector.setSearchMethod (kdTree);
        issDetector.setSalientRadius (6 * modelResolution);
        issDetector.setNonMaxRadius (4 * modelResolution);
        issDetector.setThreshold21 (0.9);
        issDetector.setThreshold32 (0.9);
        issDetector.setMinNeighbors (5);
        issDetector.setNumberOfThreads (4);
        issDetector.setInputCloud (points);
        issDetector.compute (*keypoints);
    }

    void Feature3d::DetectNARFKeypoints(pcl::PointCloud<pcl::PointXYZ>::Ptr &points,
                                        pcl::PointCloud<pcl::PointXYZ>::Ptr &keypoints)
    {
//        shared_ptr<RangeImage> pRangeImage(new RangeImage);
//        RangeImage& rangeImage = *pRangeImage;
//        float noiseLevel = 0.0;
//        float minRange = 0.0f;
//        int borderSize = 1;
//        float angularResolution = pcl::deg2rad(0.5f);//角坐标分辨率
//        Eigen::Affine3f sceneSensorPose (Eigen::Affine3f::Identity ());
//        sceneSensorPose = Eigen::Affine3f (Eigen::Translation3f (points->sensor_origin_[0],
//                                                                   points->sensor_origin_[1],
//                                                                   points->sensor_origin_[2])) *
//                            Eigen::Affine3f (points->sensor_orientation_);
//        float supportSize = 0.2f;
//
//        pcl::RangeImage::CoordinateFrame coordinateFrame = pcl::RangeImage::CAMERA_FRAME;
//        rangeImage.createFromPointCloud(points, angularResolution, pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),
//                                        sceneSensorPose, coordinateFrame, noiseLevel, minRange, borderSize);
//
//        RangeImageBorderExtractor borderExtractor;
//        NarfKeypoint narfDetector(&borderExtractor);
//        narfDetector.setRangeImage(&rangeImage);
//        narfDetector.getParameters().support_size = supportSize;
//
//        PointCloud<int> keypointIndices;
//        narfDetector.compute(keypointIndices);
//        keypoints->resize(keypointIndices.points.size());
//        for (size_t i=0; i<keypointIndices.points.size (); ++i)//按照索引获得　关键点
//            keypoints->points[i].getVector3fMap () = rangeImage.points[keypointIndices.points[i]].getVector3fMap ();
    }
}






