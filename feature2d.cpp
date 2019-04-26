//
// Created by leo on 19-4-24.
//

#include "feature2d.h"

namespace Feature2d
{

    void Feature2d::Run(string filePath, Method method)
    {
        Mat graySrc = imread(filePath, IMREAD_GRAYSCALE);
        Mat colorSrc = imread(filePath);
        Mat copySrc = imread(filePath);
        vector<Point2f> corners;
        vector<KeyPoint> keypoints;
        switch(method)
        {
            case Harris:
                cout << "Using Harris detector" << endl;
                DetectHarrisKeypoints(graySrc, corners);
                for (auto point:corners)
                    circle(copySrc, point, 2, cv::Scalar(0, 255, 0), 2, 8, 0);
                FindSubKeypoints(graySrc, corners);
                for (auto point:corners)
                    circle(copySrc, point, 3, cv::Scalar(0, 0, 255), 1, 8, 0);
                namedWindow("Harris",0);
                imshow("Harris",copySrc);
                while(1){ if(waitKey(100)==27)break; }
                break;

            case SIFT:
                cout<< "Using SIFT detector" << endl;
                DetectSIFTKeypoints(colorSrc, keypoints);
                drawKeypoints(copySrc,keypoints,copySrc);
                namedWindow("SIFT",0);
                imshow("SIFT",copySrc);
                while(1){ if(waitKey(100)==27)break; }
                break;

            case SURF:
                cout<< "Using SURF detector" << endl;
                DetectSURFKeypoints(colorSrc, keypoints);
                drawKeypoints(copySrc,keypoints,copySrc);
                namedWindow("SURF",0);
                imshow("SURF",copySrc);
                while(1){ if(waitKey(100)==27)break; }
                break;


            case ORB:
                cout<< "Using ORB detector" << endl;
                DetectORBKeypoints(colorSrc, keypoints);
                drawKeypoints(copySrc,keypoints,copySrc);
                namedWindow("ORB",0);
                imshow("ORB",copySrc);
                while(1){ if(waitKey(100)==27)break; }
                break;
            default:
                cout<<" There is no this kind of type"<<endl;
        }
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
    void Feature2d::DetectSIFTKeypoints(cv::Mat &colorSrc, std::vector<KeyPoint> &keypoints)
    {
        Ptr<Feature2D> siftDetector = xfeatures2d::SIFT::create(100);
        siftDetector->detect(colorSrc,keypoints);
    }

    //SUFT
    void Feature2d::DetectSURFKeypoints(cv::Mat &colorSrc, std::vector<cv::KeyPoint> &keypoints)
    {
        Ptr<Feature2D> surfDetector = xfeatures2d::SURF::create(100);
        surfDetector->detect(colorSrc,keypoints);
    }

    void Feature2d::DetectORBKeypoints(cv::Mat &colorSrc, vector<KeyPoint> &keypoints)
    {
        Ptr<Feature2D> orbDetector = ORB::create();
        Mat desctriptor;
        orbDetector->detectAndCompute(colorSrc,Mat(),keypoints,desctriptor);
    }
}