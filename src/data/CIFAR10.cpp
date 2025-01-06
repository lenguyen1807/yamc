#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "data/CIFAR10.h"
#include "matrix.h"

namespace fs = std::filesystem;

CIFAR10Data::CIFAR10Data(bool train)
{
  // first load label from csv file
  load_label();

  // then load image for training and testing
  // TODO: Implement this in multithread for faster performance
  load_train_im();
  load_test_im();
}

void CIFAR10Data::load_label()
{
  auto path = fs::path(std::string(CIFAR_DIR) + "labels_img.csv");
  if (!fs::is_regular_file(path)) {
    return;
  }
  std::ifstream open_file(path);
  std::string line;

  while (std::getline(open_file, line)) {
    std::stringstream ss(line);
    std::string cell;

    int idx = -1;
    size_t label;
    std::string img_name;
    while (std::getline(ss, cell, ',')) {
      if (idx == -1) {  // label
        img_name = cell;
      } else {
        label = std::stoi(cell);
      }
      idx++;
    }
    label_map[img_name] = label;
  }
}

void CIFAR10Data::load_test_im()
{
  load_im(std::string(CIFAR_DIR) + "test", test_set);
}

void CIFAR10Data::load_train_im()
{
  load_im(std::string(CIFAR_DIR) + "train", train_set);
}

void CIFAR10Data::load_im(const std::string& path,
                          std::vector<std::unique_ptr<image>>& imgs)
{
  auto fs_path = fs::path(path);
  if (!fs::is_directory(fs_path)) {
    return;
  }

  for (const auto& file : fs::directory_iterator(fs_path)) {
    image img;

    // get label
    size_t file_label = label_map.at(file.path().filename());
    auto label =
        nn::matrix<float>::onehot(file_label, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    // then load image to opencv matrix
    cv::Mat data = cv::imread(file.path(), cv::IMREAD_COLOR);
    if (data.empty()) {
      std::cout << "Could not read the image: " << file.path() << std::endl;
      return;
    }

    // normalize image to [0, 1] range
    cv::normalize(data, img.data, 0, 1, cv::NORM_MINMAX);

    img.label = label;
    // add image to vector
    imgs.emplace_back(std::make_unique<image>(img));
  }
}