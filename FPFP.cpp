#include <pcl/io/pcd_io.h>
#include <ctime>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <boost/thread/thread.hpp>
#include <pcl/features/fpfh_omp.h> //包含fpfh加速计算的omp(多核并行计算)
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_features.h> //特征的错误对应关系去除
#include <pcl/registration/correspondence_rejection_sample_consensus.h> //随机采样一致性去除
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>   //可视化

using namespace std;
typedef pcl::PointCloud<pcl::PointXYZ> pointcloud;
typedef pcl::PointCloud<pcl::Normal> pointnormal;
typedef pcl::PointCloud<pcl::FPFHSignature33> fpfhFeature;

fpfhFeature::Ptr compute_fpfh_feature(pointcloud::Ptr input_cloud, pcl::search::KdTree<pcl::PointXYZ>::Ptr tree)
{
    //法向量
    pointnormal::Ptr point_normal(new pointnormal);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> est_normal;
    est_normal.setInputCloud(input_cloud);
    est_normal.setSearchMethod(tree);
    est_normal.setKSearch(10);
    est_normal.compute(*point_normal);
    //fpfh 估计
    fpfhFeature::Ptr fpfh(new fpfhFeature);
    //pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> est_target_fpfh;
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> est_fpfh;
    est_fpfh.setNumberOfThreads(4); //指定4核计算
    est_fpfh.setInputCloud(input_cloud);
    est_fpfh.setInputNormals(point_normal);
    est_fpfh.setSearchMethod(tree);
    est_fpfh.setKSearch(10);
    est_fpfh.compute(*fpfh);

    return fpfh;

}

int main(int argc, char **argv)
{
    argc = 3;
    argv[1] = "PCDdata/bunny.pcd";
    argv[2] = "PCDdata/bunny_change.pcd";
    clock_t start, end, end2, end3;


    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    pointcloud::Ptr source(new pointcloud);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *source) == -1) // load the file
    {
        pcl::console::print_error("Couldn't read file %s!\n", argv[1]);
        return (-1);
    }
    pointcloud::Ptr target(new pointcloud);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[2], *target) == -1) // load the file
    {
        pcl::console::print_error("Couldn't read file %s!\n", argv[2]);
        return (-1);
    }

    float s1 = 0.01, s2 = 0.01;

    pointcloud::Ptr source_f(new pointcloud);
    pcl::console::print_info("Waiting for filtering the data．．．\n");
    pcl::VoxelGrid<pcl::PointXYZ> sor;    //创建滤波器对象
    sor.setInputCloud(source);                           //设置待滤波的点云
    sor.setLeafSize(s1, s1, s1);
    sor.filter(*source_f);                                 //存储

    pointcloud::Ptr target_f(new pointcloud);
    pcl::console::print_info("Waiting for filtering the data．．．\n");
    pcl::VoxelGrid<pcl::PointXYZ> sor2;    //创建滤波器对象
    sor2.setInputCloud(target);                           //设置待滤波的点云
    sor2.setLeafSize(s2, s2, s2);
    sor2.filter(*target_f);                                 //存储

    start = clock();

    fpfhFeature::Ptr source_fpfh = compute_fpfh_feature(source_f, tree);
    end3 = clock();
    cout << "calculate time is: " << float(end3 - start) / CLOCKS_PER_SEC << endl;
    cout << "source cloud number is " << source_f->points.size() << endl;

    fpfhFeature::Ptr target_fpfh = compute_fpfh_feature(target_f, tree);
    end2 = clock();
    cout << "calculate time is: " << float(end2 - end3) / CLOCKS_PER_SEC << endl;
    cout << "targrt cloud number is " << target_f->points.size() << endl;

    //对齐(占用了大部分运行时间)
    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
    sac_ia.setMaximumIterations(20);
    sac_ia.setInputSource(source_f);
    sac_ia.setSourceFeatures(source_fpfh);
    sac_ia.setInputTarget(target_f);
    sac_ia.setTargetFeatures(target_fpfh);
    pointcloud::Ptr align(new pointcloud);
    sac_ia.setNumberOfSamples(50);  //设置每次迭代计算中使用的样本数量（可省）,可节省时间
    sac_ia.align(*align);
    end = clock();
    cout << "calculate time is: " << float(end - end2) / CLOCKS_PER_SEC << endl;

    //查看点云
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor(0.0, 0.0, 0.0);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> alignColorHandler(align, 255, 0, 0);
    viewer.addPointCloud(align, alignColorHandler, "align");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "align");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> sourceColorHandler(source, 255, 0, 0);
    viewer.addPointCloud(source, sourceColorHandler, "source");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "source");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> targetColorHandler(target, 0, 255, 0);
    viewer.addPointCloud(target, targetColorHandler, "target");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target");

    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }
}
