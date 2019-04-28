#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>     //点云类型头文件
#include <pcl/correspondence.h>   //对应表示两个实体之间的匹配（例如，点，描述符等）。
#include <pcl/features/normal_3d_omp.h>   //法线
#include <pcl/features/shot_omp.h>    //描述子
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>   //均匀采样
#include <pcl/recognition/cg/hough_3d.h>    //hough算子
#include <pcl/recognition/cg/geometric_consistency.h>  //几何一致性
#include <pcl/visualization/pcl_visualizer.h>   //可视化
#include <pcl/kdtree/kdtree_flann.h>             //配准方法
#include <pcl/kdtree/impl/kdtree_flann.hpp>      //
#include <pcl/common/transforms.h>             //转换矩阵
#include <pcl/console/parse.h>
#include <pcl/search/impl/search.hpp>
#include <cmath>

typedef pcl::PointXYZRGBA PointType;  //PointXYZRGBA数据结构
typedef pcl::Normal NormalType;       //法线结构
typedef pcl::ReferenceFrame RFType;    //参考帧
typedef pcl::SHOT352 DescriptorType;   //SHOT特征的数据结构

std::string model_filename_;   //模型的文件名
std::string scene_filename_;

//Algorithm params
bool show_keypoints_(false);
bool show_correspondences_(false);
bool use_cloud_resolution_(false);
bool use_hough_(true);
float model_ss_(0.01f);
float scene_ss_(0.03f);
float rf_rad_(0.015f);
float descr_rad_(0.02f);
float cg_size_(0.01f);
float cg_thresh_(5.0f);

void showHelp(char *filename)
{
    std::cout << std::endl;
    std::cout << "***************************************************************************" << std::endl;
    std::cout << "*                                                                         *" << std::endl;
    std::cout << "*             Correspondence Grouping Tutorial - Usage Guide              *" << std::endl;
    std::cout << "*                                                                         *" << std::endl;
    std::cout << "***************************************************************************" << std::endl
              << std::endl;
    std::cout << "Usage: " << filename << " model_filename.pcd scene_filename.pcd [Options]" << std::endl << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "     -h:                     Show this help." << std::endl;
    std::cout << "     -k:                     Show used keypoints." << std::endl;
    std::cout << "     -c:                     Show used correspondences." << std::endl;
    std::cout << "     -r:                     Compute the model cloud resolution and multiply" << std::endl;
    std::cout << "                             each radius given by that value." << std::endl;
    std::cout << "     --algorithm (Hough|GC): Clustering algorithm used (default Hough)." << std::endl;
    std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;
    std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;
    std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
    std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
    std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
    std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
}

void parseCommandLine(int argc, char *argv[])
{
    //Show help
    if (pcl::console::find_switch(argc, argv, "-h"))
    {
        showHelp(argv[0]);
        exit(0);
    }

    //Model & scene filenames
    std::vector<int> filenames;
    filenames = pcl::console::parse_file_extension_argument(argc, argv, ".pcd");
    if (filenames.size() != 2)
    {
        std::cout << "Filenames missing.\n";
        showHelp(argv[0]);
        exit(-1);
    }

    model_filename_ = argv[filenames[0]];
    scene_filename_ = argv[filenames[1]];

    //Program behavior
    if (pcl::console::find_switch(argc, argv, "-k"))
    {
        show_keypoints_ = true;
    }
    if (pcl::console::find_switch(argc, argv, "-c"))
    {
        show_correspondences_ = true;
    }
    if (pcl::console::find_switch(argc, argv, "-r")) //计算点云的分辨率和多样性
    {
        use_cloud_resolution_ = true;
    }
    show_keypoints_ = true;
    show_correspondences_ = true;

    std::string used_algorithm;
    if (pcl::console::parse_argument(argc, argv, "--algorithm", used_algorithm) != -1)
    {
        if (used_algorithm.compare("Hough") == 0)
        {
            use_hough_ = true;
        } else if (used_algorithm.compare("GC") == 0)
        {
            use_hough_ = false;
        } else
        {
            std::cout << "Wrong algorithm name.\n";
            showHelp(argv[0]);
            exit(-1);
        }
    }

    //General parameters
    pcl::console::parse_argument(argc, argv, "--model_ss", model_ss_);
    pcl::console::parse_argument(argc, argv, "--scene_ss", scene_ss_);
    pcl::console::parse_argument(argc, argv, "--rf_rad", rf_rad_);
    pcl::console::parse_argument(argc, argv, "--descr_rad", descr_rad_);
    pcl::console::parse_argument(argc, argv, "--cg_size", cg_size_);
    pcl::console::parse_argument(argc, argv, "--cg_thresh", cg_thresh_);
}

double computeCloudResolution(const pcl::PointCloud<PointType>::ConstPtr &cloud)//计算点云分辨率
{
    double res = 0.0;
    int n_points = 0;
    int nres;
    std::vector<int> indices(2);
    std::vector<float> sqr_distances(2);
    pcl::search::KdTree<PointType> tree;
    tree.setInputCloud(cloud);   //设置输入点云

    for (size_t i = 0; i < cloud->size(); ++i)
    {
        if (!std::isfinite((*cloud)[i].x))
        {
            continue;
        }
        //Considering the second neighbor since the first is the point itself.
        //运算第二个临近点，因为第一个点是它本身
        nres = tree.nearestKSearch(i, 2, indices, sqr_distances);//return :number of neighbors found
        if (nres == 2)
        {
            res += sqrt(sqr_distances[1]);
            ++n_points;
        }
    }
    if (n_points != 0)
    {
        res /= n_points;
    }
    return res;
}

int main(int argc, char *argv[])
{
    argc = 3;
    argv[1] = "PCDdata/bunny.pcd";
    argv[2] = "PCDdata/bunny_change.pcd";
    argv[3] = "-k";
    parseCommandLine(argc, argv);

    pcl::PointCloud<PointType>::Ptr model(new pcl::PointCloud<PointType>()); //模型点云
    pcl::PointCloud<PointType>::Ptr model_keypoints(new pcl::PointCloud<PointType>());//模型角点
    pcl::PointCloud<PointType>::Ptr scene(new pcl::PointCloud<PointType>());  //目标点云
    pcl::PointCloud<PointType>::Ptr scene_keypoints(new pcl::PointCloud<PointType>());//目标角点
    pcl::PointCloud<NormalType>::Ptr model_normals(new pcl::PointCloud<NormalType>()); //法线
    pcl::PointCloud<NormalType>::Ptr scene_normals(new pcl::PointCloud<NormalType>());
    pcl::PointCloud<DescriptorType>::Ptr model_descriptors(new pcl::PointCloud<DescriptorType>()); //描述子
    pcl::PointCloud<DescriptorType>::Ptr scene_descriptors(new pcl::PointCloud<DescriptorType>());

    //载入文件
    if (pcl::io::loadPCDFile(model_filename_, *model) < 0)
    {
        std::cout << "Error loading model cloud." << std::endl;
        showHelp(argv[0]);
        return (-1);
    }
    if (pcl::io::loadPCDFile(scene_filename_, *scene) < 0)
    {
        std::cout << "Error loading scene cloud." << std::endl;
        showHelp(argv[0]);
        return (-1);
    }


    // 设置分辨率
    if (use_cloud_resolution_)
    {
        float resolution = static_cast<float> (computeCloudResolution(model));
        if (resolution != 0.0f)
        {
            model_ss_ *= resolution;
            scene_ss_ *= resolution;
            rf_rad_ *= resolution;
            descr_rad_ *= resolution;
            cg_size_ *= resolution;
        }

        std::cout << "Model resolution:       " << resolution << std::endl;
        std::cout << "Model sampling size:    " << model_ss_ << std::endl;
        std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
        std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
        std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
        std::cout << "Clustering bin size:    " << cg_size_ << std::endl << std::endl;
    }

    //计算法线 normalestimationomp估计局部表面特性在每个三个特点，分别表面的法向量和曲率，平行，使用OpenMP标准。//初始化调度程序并设置要使用的线程数（默认为0）。
    pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
    norm_est.setKSearch(10);       //设置K邻域搜索阀值为10个点
    norm_est.setInputCloud(model);  //设置输入点云
    norm_est.compute(*model_normals);   //计算点云法线

    norm_est.setInputCloud(scene);
    norm_est.compute(*scene_normals);

    //均匀采样点云并提取关键点
    pcl::UniformSampling<PointType> uniform_sampling;
    uniform_sampling.setInputCloud(model);  //输入点云
    uniform_sampling.setRadiusSearch(model_ss_);   //设置半径
    uniform_sampling.filter(*model_keypoints);   //滤波
    std::cout << "Model total points: " << model->size() << "; Selected Keypoints: " << model_keypoints->size()
              << std::endl;

    uniform_sampling.setInputCloud(scene);
    uniform_sampling.setRadiusSearch(scene_ss_);
    uniform_sampling.filter(*scene_keypoints);
    std::cout << "Scene total points: " << scene->size() << "; Selected Keypoints: " << scene_keypoints->size()
              << std::endl;


    //为关键点计算描述子
    pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
    descr_est.setRadiusSearch(descr_rad_);  //设置搜索半径

    descr_est.setInputCloud(model_keypoints);  //输入模型的关键点
    descr_est.setInputNormals(model_normals);  //输入模型的法线
    descr_est.setSearchSurface(model);         //输入的点云
    descr_est.compute(*model_descriptors);     //计算描述子

    descr_est.setInputCloud(scene_keypoints);  //同理
    descr_est.setInputNormals(scene_normals);
    descr_est.setSearchSurface(scene);
    descr_est.compute(*scene_descriptors);

    //  使用Kdtree找出 Model-Scene 匹配点
    pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

    pcl::KdTreeFLANN<DescriptorType> match_search;   //设置配准的方法
    match_search.setInputCloud(model_descriptors);  //输入模板点云的描述子

    //  每一个场景的关键点描述子都要找到模板中匹配的关键点描述子并将其添加到对应的匹配向量中。
    for (size_t i = 0; i < scene_descriptors->size(); ++i)
    {
        std::vector<int> neigh_indices(1);   //设置最近邻点的索引
        std::vector<float> neigh_sqr_dists(1); //申明最近邻平方距离值
        if (!std::isfinite(scene_descriptors->at(i).descriptor[0])) //忽略 NaNs点
        {
            continue;
        }
        int found_neighs = match_search.nearestKSearch(scene_descriptors->at(i), 1, neigh_indices, neigh_sqr_dists);
        //scene_descriptors->at (i)是给定点云 1是临近点个数 ，neigh_indices临近点的索引  neigh_sqr_dists是与临近点的索引

        if (found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) // 仅当描述子与临近点的平方距离小于0.25（描述子与临近的距离在一般在0到1之间）才添加匹配
        {
        //neigh_indices[0]给定点，  i  是配准数  neigh_sqr_dists[0]与临近点的平方距离
            pcl::Correspondence corr(neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
            model_scene_corrs->push_back(corr);   //把配准的点存储在容器中
        }
    }
    std::cout << "Correspondences found: " << model_scene_corrs->size() << std::endl;


    //  实际的配准方法的实现
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
    std::vector<pcl::Correspondences> clustered_corrs;

    //  使用 Hough3D算法寻找匹配点
    if (use_hough_)
    {
        //
        //  Compute (Keypoints) Reference Frames only for Hough
        //计算参考帧的Hough（也就是关键点）
        pcl::PointCloud<RFType>::Ptr model_rf(new pcl::PointCloud<RFType>());
        pcl::PointCloud<RFType>::Ptr scene_rf(new pcl::PointCloud<RFType>());
        //特征估计的方法（点云，法线，参考帧）
        pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
        rf_est.setFindHoles(true);
        rf_est.setRadiusSearch(rf_rad_);   //设置搜索半径

        rf_est.setInputCloud(model_keypoints);  //模型关键点
        rf_est.setInputNormals(model_normals); //模型法线
        rf_est.setSearchSurface(model);    //模型
        rf_est.compute(*model_rf);      //模型的参考帧

        rf_est.setInputCloud(scene_keypoints);  //同理
        rf_est.setInputNormals(scene_normals);
        rf_est.setSearchSurface(scene);
        rf_est.compute(*scene_rf);

        //  Clustering聚类的方法

        //对输入点与的聚类，以区分不同的实例的场景中的模型
        pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
        clusterer.setHoughBinSize(cg_size_);//霍夫空间设置每个bin的大小
        clusterer.setHoughThreshold(cg_thresh_);//阀值
        clusterer.setUseInterpolation(true);
        clusterer.setUseDistanceWeight(false);

        clusterer.setInputCloud(model_keypoints);
        clusterer.setInputRf(model_rf);   //设置输入的参考帧
        clusterer.setSceneCloud(scene_keypoints);
        clusterer.setSceneRf(scene_rf);
        clusterer.setModelSceneCorrespondences(model_scene_corrs);//model_scene_corrs存储配准的点

        //clusterer.cluster (clustered_corrs);辨认出聚类的对象
        clusterer.recognize(rototranslations, clustered_corrs);
    }
    else // Using GeometricConsistency  或者使用几何一致性性质
    {
        pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
        gc_clusterer.setGCSize(cg_size_);   //设置几何一致性的大小
        gc_clusterer.setGCThreshold(cg_thresh_); //阀值

        gc_clusterer.setInputCloud(model_keypoints);
        gc_clusterer.setSceneCloud(scene_keypoints);
        gc_clusterer.setModelSceneCorrespondences(model_scene_corrs);

        //gc_clusterer.cluster (clustered_corrs);
        gc_clusterer.recognize(rototranslations, clustered_corrs);
    }

    //输出的结果  找出输入模型是否在场景中出现
    std::cout << "Model instances found: " << rototranslations.size() << std::endl;
    for (size_t i = 0; i < rototranslations.size(); ++i)
    {
        std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
        std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size() << std::endl;

        // 打印处相对于输入模型的旋转矩阵与平移矩阵
        Eigen::Matrix3f rotation = rototranslations[i].block<3, 3>(0, 0);
        Eigen::Vector3f translation = rototranslations[i].block<3, 1>(0, 3);

        printf("\n");
        printf("            | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1), rotation(0, 2));
        printf("        R = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1), rotation(1, 2));
        printf("            | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1), rotation(2, 2));
        printf("\n");
        printf("        t = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1), translation(2));
    }

    //可视化
    pcl::visualization::PCLVisualizer viewer("Correspondence Grouping");
    viewer.addPointCloud(scene, "scene_cloud");//可视化场景点云

    pcl::PointCloud<PointType>::Ptr off_scene_model(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints(new pcl::PointCloud<PointType>());

    if (show_correspondences_ || show_keypoints_)  //可视化配准点
    {
        //  We are translating the model so that it doesn't end in the middle of the scene representation
        //就是要对输入的模型进行旋转与平移，使其在可视化界面的中间位置
        pcl::transformPointCloud(*model, *off_scene_model, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));
        pcl::transformPointCloud(*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f(-1, 0, 0),
                                 Eigen::Quaternionf(1, 0, 0, 0));  //因为模型的位置变化了，所以要对特征点进行变化

        pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler(off_scene_model, 255,
                                                                                                  255, 128);  //对模型点云上颜色
        viewer.addPointCloud(off_scene_model, off_scene_model_color_handler, "off_scene_model");
    }

    if (show_keypoints_)   //可视化关键点
    {
        pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler(scene_keypoints, 0, 0,
                                                                                                  255);  //对场景中的点云上为绿色
        viewer.addPointCloud(scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

        pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler(
                off_scene_model_keypoints, 0, 0, 255);
        viewer.addPointCloud(off_scene_model_keypoints, off_scene_model_keypoints_color_handler,
                             "off_scene_model_keypoints");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5,
                                                "off_scene_model_keypoints");
    }

    for (size_t i = 0; i < rototranslations.size(); ++i)
    {
        pcl::PointCloud<PointType>::Ptr rotated_model(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*model, *rotated_model, rototranslations[i]);//把model转化为rotated_model
        //    <Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >   rototranslations是射影变换矩阵
        std::stringstream ss_cloud;
        ss_cloud << "instance" << i;

        pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler(rotated_model, 255, 0,
                                                                                                0);
        viewer.addPointCloud(rotated_model, rotated_model_color_handler, ss_cloud.str());

        if (show_correspondences_)   //显示配准连接
        {
            for (size_t j = 0; j < clustered_corrs[i].size(); ++j)
            {
                std::stringstream ss_line;
                ss_line << "correspondence_line" << i << "_" << j;
                PointType &model_point = off_scene_model_keypoints->at(clustered_corrs[i][j].index_query);
                PointType &scene_point = scene_keypoints->at(clustered_corrs[i][j].index_match);

                //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
                viewer.addLine<PointType, PointType>(model_point, scene_point, 0, 255, 0, ss_line.str());
            }
        }
    }

    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }

    return (0);
}