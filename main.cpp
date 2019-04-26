#include <iostream>
#include "feature3d.h"
#include "feature2d.h"


int main() {
    Feature2d::Feature2d feature2d;
    Feature3d::Feature3d feature3d;
    feature3d.Run("PCDdata/bunny.pcd", Feature3d::ISS);
    //feature2d.Run("1.jpg",Feature2d::ORB);
}