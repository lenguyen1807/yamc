# Neural Networks from Scratch in C++

## How to run

- Make sure have installed `CMake`.
- Make directory `build` and:
```bash
cd build
cmake ..
make
```

## Usage

- Change `EPOCHS` parameter in `utils.h` (default to 1).
- Can implement more optimizer in `nn.cpp` or implement more loss function in `nn.cpp` too.

## Result

<img src="./result.png" alt="Result image">

- Train with only 1 epoch.
- Test accuracy: 93%.
- Used memory: 1.6 GB.
- Train on 60000 MNIST images and test on 10000 MNIST images, data in `archives.zip` in `data` folder.
- Neural Network has 3 hidden layers. Used SGD (Stochastic Gradient Descent) for learning process.

## What can be done

- [x] Use SGD (Stochastic Gradient Descent) for opitmization algorithm.
- [ ] Implement mini-batch.
- [ ] Implement momentum to SGD.
- [ ] Implement more optimization algorithm (AdaGrad, AdamW).
- [ ] Implement Dropout (prevent overfitting).
- [ ] Implement Batch Normalization.
- [ ] Use GPU.

## Resources

- https://github.com/lionelmessi6410/Neural-Networks-from-Scratch/tree/main?tab=readme-ov-file