> NOTE: This code has been test only on MacOS (but maybe can work on Linux, remember I do not support Windows so you should know what to do if you want to run this on Windows).

You need to install `OpenMP` and `OpenCV`. Using `brew` as below:
```cpp
brew install llvm
brew install libomp
brew install opencv
```

> TODO: Implement tutorial

**TL'DR**: This is the (another) rewrite branch of my old source (few months ago), the old version took about 37 minutes to finish training and testing when the new version only took 9 minutes (insane improvement), but all of these are on my old machine. On my new machine (Macbook Air M2), the performance now is 2 minutes 💀. I also changed the code structure and add **dropout**, **CNN** layer.

### What this code has

- **Layer**: Convolution, Linear, Dropout, Maxpooling, AvgPooling, Local Response Normalization.
- **Model**: MLP, AlexNet, VGG-16, LeNet5.
- **Activation**: ReLU, LeakyReLU, Softmax.
- **Loss function**: Cross Entropy Loss.
- **Optimizer**: SGD (Stochastic Gradient Descent).
- Flexible code structures. But because of my machine VGG-16 and AlexNet can do inference only (do not have enough memory for training and also my implementation is not optimized well).
- Below is example of VGG-16:

```cpp
VGG16::VGG16(size_t input_channels, size_t output_size)
{
  /* First CNN Layer */
  add<0, nn::Conv2D>(input_channels,
                     /*output_channels=*/64,
                     /*stride=*/1,
                     /*kernel_size=*/3,
                     /*padding=*/1);
  add<1, nn::ReLU>();
  add<2, nn::Conv2D>(/*input_channels=*/64,
                     /*output_channels=*/64,
                     /*stride=*/1,
                     /*kernel_size=*/3,
                     /*padding=*/1);
  add<3, nn::ReLU>();
  add<4, nn::Maxpool2D>(/*kernel_size=*/2,
                        /*stride=*/2,
                        /*padding=*/0);

  /* Second CNN Layer */
  add<5, nn::Conv2D>(/*input_channels=*/64,
                     /*output_channels=*/128,
                     /*stride=*/1,
                     /*kernel_size=*/3,
                     /*padding=*/1);
  add<6, nn::ReLU>();
  add<7, nn::Conv2D>(/*input_channels=*/128,
                     /*output_channels=*/128,
                     /*stride=*/1,
                     /*kernel_size=*/3,
                     /*padding=*/1);
  add<8, nn::ReLU>();
  add<9, nn::Maxpool2D>(/*kernel_size=*/2,
                        /*stride=*/2,
                        /*padding=*/0);

  /* Third CNN Layer */
  add<10, nn::Conv2D>(/*input_channels=*/128,
                      /*output_channels=*/256,
                      /*stride=*/1,
                      /*kernel_size=*/2,
                      /*padding=*/1);
  add<11, nn::ReLU>();
  add<12, nn::Conv2D>(/*input_channels=*/256,
                      /*output_channels=*/256,
                      /*stride=*/1,
                      /*kernel_size=*/3,
                      /*padding=*/1);
  add<13, nn::ReLU>();
  add<14, nn::Maxpool2D>(/*kernel_size=*/2,
                         /*stride=*/2,
                         /*padding=*/0);

  /* Fourth CNN Layer */
  add<15, nn::Conv2D>(/*input_channels=*/256,
                      /*output_channels=*/512,
                      /*stride=*/1,
                      /*kernel_size=*/2,
                      /*padding=*/1);
  add<16, nn::ReLU>();
  add<17, nn::Conv2D>(/*input_channels=*/512,
                      /*output_channels=*/512,
                      /*stride=*/1,
                      /*kernel_size=*/3,
                      /*padding=*/1);
  add<18, nn::ReLU>();
  add<19, nn::Maxpool2D>(/*kernel_size=*/2,
                         /*stride=*/2,
                         /*padding=*/0);

  /* Flatten */
  add<20, nn::Flatten>();

  /* Classifer */
  add<21, nn::Dropout>();
  add<22, nn::Linear>(2 * 2 * 512, 4096);
  add<23, nn::ReLU>();
  add<24, nn::Dropout>();
  add<25, nn::Linear>(4096, 4096);
  add<26, nn::ReLU>();
  add<27, nn::Linear>(4096, output_size);
}

nn::matrix<float> VGG16::forward(const cv::Mat& image)
{
  /* First CNN Layer */
  cv::Mat im = layers.at(0)->forward(image);  // Conv2D
  im = layers.at(1)->forward(im);  // ReLU
  im = layers.at(2)->forward(im);  // Conv2D
  im = layers.at(3)->forward(im);  // ReLU
  im = layers.at(4)->forward(im);  // Maxpool2D

  /* Second CNN Layer */
  im = layers.at(5)->forward(im);  // Conv2D
  im = layers.at(6)->forward(im);  // ReLU
  im = layers.at(7)->forward(im);  // Conv2D
  im = layers.at(8)->forward(im);  // ReLU
  im = layers.at(9)->forward(im);  // Maxpool2D

  /* Third CNN Layer */
  im = layers.at(10)->forward(im);  // Conv2D
  im = layers.at(11)->forward(im);  // ReLU
  im = layers.at(12)->forward(im);  // Conv2D
  im = layers.at(13)->forward(im);  // ReLU
  im = layers.at(14)->forward(im);  // Maxpool2D

  /* Fourth CNN Layer */
  im = layers.at(15)->forward(im);  // Conv2D
  im = layers.at(16)->forward(im);  // ReLU
  im = layers.at(17)->forward(im);  // Conv2D
  im = layers.at(18)->forward(im);  // ReLU
  im = layers.at(19)->forward(im);  // Maxpool2D

  /* Flatten */
  nn::matrix<float> result =
      dynamic_cast<nn::Flatten*>(layers.at(20).get())->forward_im(im);

  /* Classifier */
  result = layers.at(21)->forward(result);  // Dropout
  result = layers.at(22)->forward(result);  // Linear
  result = layers.at(23)->forward(result);  // ReLU
  result = layers.at(24)->forward(result);  // Dropout
  result = layers.at(25)->forward(result);  // Linear
  result = layers.at(26)->forward(result);  // ReLU
  result = layers.at(27)->forward(result);  // Linear

  return result;
}

void VGG16::backward(const nn::matrix<float>& grad)
{
  /* Backward Classifier */
  nn::matrix<float> result = layers.at(27)->backward(grad);  // Linear
  result = layers.at(26)->forward(result);  // ReLU
  result = layers.at(25)->forward(result);  // Linear
  result = layers.at(24)->forward(result);  // Dropout
  result = layers.at(23)->forward(result);  // ReLU
  result = layers.at(22)->forward(result);  // Linear
  result = layers.at(21)->forward(result);  // Dropout

  /* Backward Flatten */
  cv::Mat im =
      dynamic_cast<nn::Flatten*>(layers.at(20).get())->backward_im(result);

  /* Fourth CNN Layer */
  im = layers.at(19)->backward(im);  // Maxpool2D
  im = layers.at(18)->backward(im);  // ReLU
  im = layers.at(17)->backward(im);  // Conv2D
  im = layers.at(16)->backward(im);  // ReLU
  im = layers.at(15)->backward(im);  // Conv2D

  /* Third CNN Layer */
  im = layers.at(14)->backward(im);  // Maxpool2D
  im = layers.at(13)->backward(im);  // ReLU
  im = layers.at(12)->backward(im);  // Conv2D
  im = layers.at(11)->backward(im);  // ReLU
  im = layers.at(10)->backward(im);  // Conv2D

  /* Second CNN Layer */
  im = layers.at(9)->backward(im);  // Maxpool2D
  im = layers.at(8)->backward(im);  // ReLU
  im = layers.at(7)->backward(im);  // Conv2D
  im = layers.at(6)->backward(im);  // ReLU
  im = layers.at(5)->backward(im);  // Conv2D

  /* First CNN Layer */
  im = layers.at(4)->backward(im);  // Maxpool2D
  im = layers.at(3)->backward(im);  // ReLU
  im = layers.at(2)->backward(im);  // Conv2D
  im = layers.at(1)->backward(im);  // ReLU
  im = layers.at(0)->backward(im);  // Conv2D
}
```

> TODO (in far far future): Rewrite the whole source and implement batch

### Results

---

<img src="results/ryzen5_5500u_windows.png">
The old version.

---

<img src="results/ryzen5_5500u_linux.png">
First rewrite version (adding BLAS).

---

<img src="results/apple_m2.png">
Second rewrite version (improve code structures and run on Mac M2).

<img src="results/m2_temp.png">
But the CPU got too hot (about 100C and I use power adapter, not on battery)

---

> Seeing more on my blog: https://lenguyen.vercel.app/projects/cpp-nn