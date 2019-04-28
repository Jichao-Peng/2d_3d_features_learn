#include <iostream>
#include "feature3d.h"
#include "feature2d.h"


int main()
{
    Feature2d::Feature2d feature2d;
    Feature3d::Feature3d feature3d;
    feature3d.Run("PCDdata/bunny.pcd", Feature3d::NARF);
    //feature2d.Run("1.jpg", "1.png", Feature2d::ORB);
}