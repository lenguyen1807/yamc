#include <fstream>
#include <sstream>
#include <string>

#include "data/MNIST.h"
#include "helper.h"

MNISTData::MNISTData(const std::string& path)
{
  std::ifstream open_file(path.c_str());
  std::string line;

  // skip first line
  std::getline(open_file, line);

  // read remain line
  while (std::getline(open_file, line)) {
    int idx = -1;

    std::stringstream ss(line);
    std::string cell;
    image img;
    img.data = nn::matrix<float>(28 * 28, 1);

    // hack to parse csv file type
    while (std::getline(ss, cell, ',')) {
      if (idx == -1) {
        img.label = nn::matrix<float>::onehot(std::stoi(cell),
                                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
      } else {
        img.data.data[idx] = std::stof(cell);
      }
      idx++;
    }

    // before emplace the image, we should normalize it to [0.0, 1.0]
    img.data = normalize_minmax(img.data);
    dataset.emplace_back(std::make_unique<image>(img));
  }
}