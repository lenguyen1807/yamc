#ifndef IMAGE_H
#define IMAGE_H

#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include "utils.h"
#include "matrix.h"

struct Image 
{
    MatrixPtr label;
    MatrixPtr data;

    Image()
    : label(0)
    , data(nullptr)
    {}
};

ImageVector ReadData(const std::string& path) {
    std::ifstream openFile(path);

    std::string line;
    ImageVector imgs;

    // skip first line
    std::getline(openFile, line);

    // read remain file
    while (std::getline(openFile, line))
    {
        int idx = -1;
        ImagePtr img = std::make_unique<Image>();
        Matrix data(28, 28);

        std::stringstream ssLine(line);
        std::string cell;

        while (std::getline(ssLine, cell, ','))
        {
            if (idx == -1) 
            {
                img->label = std::make_unique<Matrix>(Matrix::OneHot(std::stoi(cell), 10));
            }
            else 
            { 
                size_t j = idx % 28;
                data.values[(idx - j)/28][j] = std::stod(cell); 
            }
            idx++;
        }

        img->data = std::make_unique<Matrix>(Matrix::Flatten(data));

        imgs.emplace_back(std::move(img));
    }

    return imgs;
}

#endif // IMAGE_H