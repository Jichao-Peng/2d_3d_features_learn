//
// Created by leo on 19-4-24.
//

#include "feature2d.h"

namespace Feature2d
{

    void Feature2d::Run(string filePath1, string filePath2, Method method)
    {
        
        Mat colorSrc1 = imread(filePath1);
        Mat copySrc1 = colorSrc1.clone();
        Mat colorSrc2 = imread(filePath2);
        Mat copySrc2 = colorSrc2.clone();

        if(method == Harris)
        {
            cout << "Using Harris detector" << endl;

            vector<Point2f> corners;
            Mat graySrc1;
            cvtColor(colorSrc1,graySrc1,COLOR_RGB2GRAY);

            DetectHarrisKeypoints(graySrc1, corners);
            for (auto point:corners)
                circle(copySrc1, point, 2, cv::Scalar(0, 255, 0), 2, 8, 0);
            FindSubKeypoints(graySrc1, corners);
            for (auto point:corners)
                circle(copySrc1, point, 3, cv::Scalar(0, 0, 255), 1, 8, 0);
            namedWindow("Harris", 0);
            imshow("Harris", copySrc1);
            while (1)
            { if (waitKey(100) == 27)break; }
        }
        else if(method == SIFT)
        {
            vector<KeyPoint> keypoints1;
            Mat descriptor1;
            vector<KeyPoint> keypoints2;
            Mat descriptor2;

            //提取特征点点和匹配的过程
            cout << "Using SIFT detector" << endl;
            DetectSIFTKeypoints(colorSrc1, keypoints1,descriptor1);
            DetectSIFTKeypoints(colorSrc2, keypoints2,descriptor2);
            vector<DMatch> matches;
            FlannBasedMatcher fnMatcher;
            fnMatcher.match(descriptor1,descriptor2,matches);

            //筛选匹配点
            double minDist = 1000, maxDist = 0;
            // 找出所有匹配之间的最大值和最小值
            for (int i = 0; i < descriptor1.rows; i++)
            {
                double dist = matches[i].distance;//汉明距离在matches中
                if (dist < minDist) minDist = dist;
                if (dist > maxDist) maxDist = dist;
            }
            // 当描述子之间的匹配大于2倍的最小距离时，即认为该匹配是一个错误的匹配。
            // 但有时描述子之间的最小距离非常小，可以设置一个经验值作为下限
            vector<DMatch> goodMatches;
            for (int i = 0; i < descriptor1.rows; i++)
            {
                if (matches[i].distance <= max(2 * minDist, 30.0))
                    goodMatches.push_back(matches[i]);
            }

            Mat result;
            drawMatches(copySrc1,keypoints1,colorSrc2,keypoints2,goodMatches,result);

            namedWindow("SIFT", 0);
            imshow("SIFT", result);
            while (1)
            { if (waitKey(100) == 27)break; }
        }
        else if(method == SURF)
        {
            vector<KeyPoint> keypoints1;
            Mat descriptor1;
            vector<KeyPoint> keypoints2;
            Mat descriptor2;

            //提取特征点点和匹配的过程
            cout << "Using SURF detector" << endl;
            DetectSURFKeypoints(colorSrc1, keypoints1,descriptor1);
            DetectSURFKeypoints(colorSrc2, keypoints2,descriptor2);
            vector<DMatch> matches;
            FlannBasedMatcher fnMatcher;
            fnMatcher.match(descriptor1,descriptor2,matches);

            //筛选匹配点
            sort(matches.begin(), matches.end()); //筛选匹配点
            vector< DMatch > goodMatches;
            int ptsPairs = std::min(50, (int)(matches.size() * 0.15));
            for (int i = 0; i < ptsPairs; i++)
            {
                goodMatches.push_back(matches[i]);
            }

            Mat result;
            drawMatches(copySrc1,keypoints1,colorSrc2,keypoints2,goodMatches,result);

            namedWindow("SURF", 0);
            imshow("SURF", result);
            while (1)
            { if (waitKey(100) == 27)break; }


            vector<KeyPoint> keypoints;
            Mat descriptor;
        }
        else if(method == ORB)
        {
            vector<KeyPoint> keypoints1;
            Mat descriptor1;
            vector<KeyPoint> keypoints2;
            Mat descriptor2;

            //提取特征点点和匹配的过程
            cout << "Using SURF detector" << endl;
            DetectORBKeypoints(colorSrc1, keypoints1, descriptor1);
            DetectORBKeypoints(colorSrc2, keypoints2, descriptor2);
            vector<DMatch> matches;
            BFMatcher bfMatcher;
            bfMatcher.match(descriptor1,descriptor2,matches);

            //筛选匹配点
            //筛选匹配点
            double dist_min = 1000; double dist_max = 0;
            for(size_t t = 0;t<matches.size();t++)
            {
                double dist = matches[t].distance;
                if(dist<dist_min) dist_min = dist;
                if(dist>dist_max) dist_max = dist;
            }
            vector<DMatch> goodMatches;
            for(size_t t = 0;t<matches.size();t++)
            {
                double dist = matches[t].distance;
                if(dist <= max(2*dist_min,30.0))
                    goodMatches.push_back(matches[t]); }


            Mat result;
            drawMatches(copySrc1,keypoints1,colorSrc2,keypoints2,goodMatches,result);

            namedWindow("ORB", 0);
            imshow("ORB", result);
            while (1)
            { if (waitKey(100) == 27)break; }


            vector<KeyPoint> keypoints;
            Mat descriptor;
        }
        else cout<<" There is no this kind of type"<<endl;
        cout<<"Detect finished"<<endl;
    }


    //Harris
    void Feature2d::DetectHarrisKeypoints(cv::Mat &graySrc, vector<Point2f> &outputCorners)
    {
        outputCorners.clear();
        int maxCorners = 25;
        double qualityLevel = 0.01;
        double minDistance = 10;
        int blockSize = 3;
        double k = 0.04;
        goodFeaturesToTrack(graySrc,//输入图像
                            outputCorners, //输出角点
                            maxCorners, //最大角点数目
                            qualityLevel, //质量水平系数
                            minDistance, //最小距离
                            Mat(), //输入一个尺度大小相同的Mask，Mask为零处不进行角点检测
                            blockSize, //使用的邻域数
                            false, //等于false则使用Shi Tomasi角点检测法，等于True时使用Harris角点检测法
                            k);// 使用Harris角点检测法时才使用
    }

    void Feature2d::FindSubKeypoints(Mat &graySrc, vector<Point2f> &outputCorners)
    {
        Size winSize = Size(5, 5);
        Size zeroZone = Size(-1, -1);
        TermCriteria stopCriteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER,//决定迭代停止的方式
                                                 40, //最大迭代次数
                                                 0.001); //特定的阈值

        cornerSubPix(graySrc, //输入图像
                     outputCorners, //输入粗角点，输出精角点
                     winSize, //搜索窗口边长的一半
                     zeroZone, //搜索窗口中间死区的边长的一半，（-1,-1）代表没有这个区域
                     stopCriteria); //终止条件
    }

    //SIFT
    void Feature2d::DetectSIFTKeypoints(cv::Mat &colorSrc, std::vector<KeyPoint> &keypoints,  Mat& descriptor)
    {
        Ptr<Feature2D> siftDetector = xfeatures2d::SIFT::create(100);
        siftDetector->detectAndCompute(colorSrc,noArray(),keypoints,descriptor);
    }

    //SUFT
    void Feature2d::DetectSURFKeypoints(cv::Mat &colorSrc, std::vector<cv::KeyPoint> &keypoints,  Mat& descriptor)
    {
        Ptr<Feature2D> surfDetector = xfeatures2d::SURF::create(100);
        surfDetector->detectAndCompute(colorSrc,noArray(),keypoints,descriptor);
    }

    void Feature2d::DetectORBKeypoints(cv::Mat &colorSrc, vector<KeyPoint> &keypoints,  Mat& descriptor)
    {
        Ptr<Feature2D> orbDetector = ORB::create(500);
        orbDetector->detectAndCompute(colorSrc,Mat(),keypoints,descriptor);
    }
}