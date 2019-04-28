//
// Created by leo on 19-4-23.
//

#ifndef CODE_FEATURE3D_H
#define CODE_FEATURE3D_H

#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/common/io.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <string.h>

namespace Feature3d
{
    using namespace pcl;
    using namespace std;

    enum Method{NARF,SIFT,Harris,ISS};

    class Feature3d
    {
    public:
        void Run(const string &filePath, Method method);


    private:
        //3D-SIFT
        void DetectSIFTKeypoints(PointCloud<PointXYZRGB>::Ptr &points, PointCloud<PointWithScale>::Ptr &keypoints);

        //3D-Harris
        void DetectHarrisKeypoints(PointCloud<PointXYZRGB>::Ptr &points, PointCloud<PointXYZI>::Ptr &keypoints);

        //ISS
        void DetectISSKeypoints(PointCloud<PointXYZRGB>::Ptr &points, PointCloud<PointXYZRGB>::Ptr &keypoints);

        //NARF
        void DetectNARFKeypoints(PointCloud<PointXYZ>::Ptr &points, PointCloud<PointXYZ>::Ptr &keypoints);
    };

}


#endif //CODE_FEATURE3D_H
