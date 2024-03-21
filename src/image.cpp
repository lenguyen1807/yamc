#include "image.h"
#include <sstream>
#include <iostream>
#include <fstream>

std::vector<Image*> Image::ReadData(const std::string& path)
{
    std::ifstream openFile(path);

    std::string line;
    std::vector<Image*> imgs;

    // skip first line
    std::getline(openFile, line);

    // read remain file
    while (std::getline(openFile, line))
    {
        size_t idx = -1;
        Image* img = new Image();

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
                img->data.values[(idx - j)/28][j] = std::stod(cell); 
            }
            idx++;
        }

        imgs.push_back(img);
    }

    openFile.close();

    return imgs;
}