#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "test_helper.h"

#ifdef HAVE_OPENCV_XFEATURES2D
#  include "opencv2/features2d.hpp"
#  include "opencv2/highgui.hpp"
#  include "opencv2/xfeatures2d.hpp"
#endif

namespace cvf = cv::xfeatures2d;

bool is_close_mat(const nn::matrix<float>& mat1,
                  const nn::matrix<float>& mat2,
                  float epsilon)
{
  if ((mat1.rows != mat2.rows) || (mat1.cols != mat2.cols)) {
    return false;
  }

  for (size_t i = 0; i < mat1.rows * mat2.cols; i++) {
    if (!are_equal<float>(mat1.data[i], mat2.data[i], epsilon)) {
      return false;
    }
  }

  return true;
}

bool is_close_im(const cv::Mat& im1, const cv::Mat& im2, float xy_threshold)
{
  if (im1.rows != im2.rows || im1.cols != im2.cols
      || im1.channels() != im2.channels())
  {
    return false;
  }

  int h = im1.rows;
  int w = im1.cols;

  // Use MSE to find distance between two image
  cv::Mat diff;
  cv::absdiff(im1, im2, diff);  // |I1 - I2|
  diff = diff.mul(diff);  // |I1 - I2|^2
  cv::Scalar s = cv::sum(diff);
  float sse = s.val[0] + s.val[1] + s.val[2];
  float mse = sse / static_cast<float>(im1.channels() * im1.total());
  return mse < xy_threshold;

  // //-- Step 1: Detect the keypoints using SURF Detector, compute the
  // descriptors int min_hessian =
  //     800;  // smaller value finds more and bigger value less features
  // cv::Ptr<cvf::SURF> detector = cvf::SURF::create(min_hessian);
  // std::vector<cv::KeyPoint> keypoints1, keypoints2;
  // cv::Mat descriptors1, descriptors2;

  // detector->detectAndCompute(im1, cv::noArray(), keypoints1, descriptors1);
  // detector->detectAndCompute(im2, cv::noArray(), keypoints2, descriptors2);

  // //-- Step 2: Matching descriptor vectors with a FLANN based matcher
  // // Since SURF is a floating-point descriptor NORM_L2 is used
  // cv::Ptr<cv::DescriptorMatcher> matcher =
  //     cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  // std::vector<cv::DMatch> knn_matches;

  // // no features images is not a screenshot
  // if (descriptors1.empty() || descriptors2.empty()) {
  //   return false;
  // }

  // matcher->match(descriptors1, descriptors2, knn_matches, 2);
  // //-- using a threshold for the distance of a match
  // const float distanceThresh = 0.5;

  // int matchCount = 0;

  // for (size_t i = 0; i < knn_matches.size(); i++) {
  //   // check if distance of match is small. check if x and y in left are
  //   right
  //   // image almost the same, keep in mind that a feature can be slightly
  //   // different even in an image which looks the same for an human
  //   if (knn_matches[i].distance < distanceThresh
  //       && abs(keypoints1[knn_matches[i].queryIdx].pt.x
  //              - keypoints2[knn_matches[i].trainIdx].pt.x)
  //           < xy_threshold
  //       && abs(keypoints1[knn_matches[i].queryIdx].pt.y
  //              - keypoints2[knn_matches[i].trainIdx].pt.y)
  //           < xy_threshold)
  //   {
  //     matchCount++;
  //   }
  // }

  // // have at least 18 of those features, this works fine for me
  // // but depending on the image you may need another value here
  // return matchCount > 18;
}