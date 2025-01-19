#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "data/TinyImageNet.h"
#include "matrix.h"

namespace fs = std::filesystem;

TinyImageNetData::TinyImageNetData(bool train)
{
  // TODO: Implement this in multithread for faster performance
  if (train) {
    load_label();
    load_train_im();
  } else {
    load_test_label();
    load_test_im();
  }
}

void TinyImageNetData::load_label()
{
  auto path = fs::path(std::string(TINY_IMAGENET_DIR) + "label_ids.csv");
  if (!fs::is_regular_file(path)) {
    return;
  }
  std::ifstream open_file(path);
  std::string line;

  while (std::getline(open_file, line)) {
    if (line.empty()) {
      continue;
    }

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

void TinyImageNetData::load_train_im()
{
  auto fs_path = fs::path(std::string(TINY_IMAGENET_DIR) + "train");
  if (!fs::is_directory(fs_path)) {
    return;
  }

  /* File structure:
  label_name
  ---- images
  -------- .... (.JPEG)
  ---- bounding boxes
  */

  nn::matrix<float> labels = nn::matrix<float>::arange(0, 200);

  for (const auto& dir : fs::directory_iterator(fs_path)) {
    // skip all file
    if (!dir.is_directory()) {
      continue;
    }

    // load label
    size_t data_label = label_map.at(dir.path().filename());
    auto label = nn::matrix<float>::onehot(data_label, labels.data);

    for (const auto& dir : fs::directory_iterator(dir.path())) {
      if (!dir.is_directory()) {
        continue;
      }

      // load image
      for (const auto& file : fs::directory_iterator(dir.path())) {
        image img;

        cv::Mat data = cv::imread(file.path(), cv::IMREAD_COLOR);
        if (data.empty()) {
          std::cout << "Could not read the image: " << file.path() << std::endl;
          return;
        }

        // We also want to resize image
        cv::resize(data, img.data, cv::Size(224, 224), cv::INTER_LINEAR);

        // normalize image to [0, 1] range
        cv::normalize(img.data,
                      img.data,
                      0,
                      1,
                      cv::NORM_MINMAX,
                      CV_32FC(data.channels()));

        img.label = label;
        // add image to vector
        dataset.emplace_back(std::make_unique<image>(img));
      }
    }
  }

  /* We need to shuffle data random before feed to Neural Network */
  auto rng = std::default_random_engine {};
  std::shuffle(dataset.begin(), dataset.end(), rng);
}

void TinyImageNetData::load_test_im()
{
  auto fs_path = fs::path(std::string(TINY_IMAGENET_DIR) + "val");
  if (!fs::is_directory(fs_path)) {
    return;
  }

  nn::matrix<float> labels = nn::matrix<float>::arange(0, 200);

  for (const auto& dir : fs::directory_iterator(fs_path)) {
    if (!dir.is_directory()) {
      continue;
    }

    // load image
    for (const auto& file : fs::directory_iterator(dir.path())) {
      image img;

      // get label
      size_t file_label = test_labels.at(file.path().filename());
      auto label = nn::matrix<float>::onehot(file_label, labels.data);

      // get data
      cv::Mat data = cv::imread(file.path(), cv::IMREAD_COLOR);
      if (data.empty()) {
        std::cout << "Could not read the image: " << file.path() << std::endl;
        return;
      }

      // We also want to resize image
      cv::resize(data, img.data, cv::Size(224, 224), cv::INTER_LINEAR);

      // normalize image to [0, 1] range
      cv::normalize(
          img.data, img.data, 0, 1, cv::NORM_MINMAX, CV_32FC(data.channels()));

      img.label = label;

      // add image to vector
      dataset.emplace_back(std::make_unique<image>(img));
    }
  }
}

void TinyImageNetData::load_test_label()
{
  auto path = fs::path(std::string(TINY_IMAGENET_DIR) + "val_labels.csv");
  if (!fs::is_regular_file(path)) {
    return;
  }

  std::ifstream open_file(path);
  std::string line;

  while (std::getline(open_file, line)) {
    if (line.empty()) {
      continue;
    }

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
    test_labels[img_name] = label;
  }
}