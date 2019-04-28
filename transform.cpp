//
// Created by leo on 19-4-28.
//

#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>


int main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ> ());

    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("PCDdata/bunny.pcd", *source_cloud) == -1)
    {
        PCL_ERROR ("Couldn't read file sample.pcd \n");
        return (-1);
    }
    Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
    float theta = M_PI/4; // The angle of rotation in radians
    transform_1 (0,0) = cos (theta);
    transform_1 (0,1) = -sin(theta);
    transform_1 (1,0) = sin (theta);
    transform_1 (1,1) = cos (theta);
    transform_1 (0,3) = 0;

    printf ("Method #1: using a Matrix4f\n");
    std::cout << transform_1 << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::transformPointCloud (*source_cloud, *transformed_cloud, transform_1);

    //Save
    pcl::PCDWriter writer;
    writer.write("PCDdata/bunny_change.pcd",*transformed_cloud);
    cout<<"Saved!!"<<endl;

    // Visualization
    printf(  "\nPoint cloud colors :  white  = original point cloud\n"
             "                        red  = transformed point cloud\n");
    pcl::visualization::PCLVisualizer viewer ("Matrix transformation example");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler (source_cloud, 255, 255, 255);
    viewer.addPointCloud (source_cloud, source_cloud_color_handler, "original_cloud");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color_handler (transformed_cloud, 230, 20, 20);
    viewer.addPointCloud (transformed_cloud, transformed_cloud_color_handler, "transformed_cloud");
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0);
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "transformed_cloud");
    while (!viewer.wasStopped ())
    {
        viewer.spinOnce ();
    }

    return 0;
}