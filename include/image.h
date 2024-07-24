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
    MatrixPtr data;
    int label;

    Image()
    : label(0)
    , data(std::make_unique<Matrix>(28, 28))
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
        size_t idx = -1;
        ImagePtr img = std::make_unique<Image>();

        std::stringstream ssLine(line);
        std::string cell;

        while (std::getline(ssLine, cell, ','))
        {
            if (idx == -1) 
            {
                img->label = std::stoi(cell); 
            }
            else 
            { 
                size_t j = idx % 28;
                img->data->values[(idx - j)/28][j] = std::stod(cell); 
            }
            idx++;
        }

        imgs.emplace_back(std::move(img));
    }

    return imgs;
}

#endif // IMAGE_H