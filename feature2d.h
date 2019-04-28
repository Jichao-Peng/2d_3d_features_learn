//
// Created by leo on 19-4-24.
//

#ifndef CODE_FEATURE2D_H
#define CODE_FEATURE2D_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>


namespace Feature2d
{
    using namespace cv;
    using namespace std;

    enum Method{Harris, SIFT, SURF, ORB};

    class Feature2d
    {
    public:
        void Run(string filePath1, string filePath2, Method method);

    private:
        //Harris
        void DetectHarrisKeypoints(Mat &graySrc, vector<Point2f> &outputCorners);
        void FindSubKeypoints(Mat &graySrc, vector<Point2f> &outputCorners);

        //SIFT
        void DetectSIFTKeypoints(Mat &colorSrc, vector<KeyPoint> &keypoints, Mat& descriptor);

        //SUFT
        void DetectSURFKeypoints(Mat &colorSrc, vector<KeyPoint> &keypoints, Mat& descriptor);

        //ORB
        void DetectORBKeypoints(Mat &colorSr, vector<KeyPoint> &keypoints, Mat& descriptor);
    };

}

#endif //CODE_FEATURE2D_H
