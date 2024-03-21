#ifndef IMAGE_H
#define IMAGE_H

#include "matrix.h"
#include <string>

class Image
{
public:
    Matrix data;
    int label;

    Image()
    : label(0)
    , data(28, 28)
    {}

    static std::vector<Image*> ReadData(const std::string& path);
};

#endif // IMAGE_H