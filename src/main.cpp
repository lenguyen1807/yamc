#include <iostream>
#include "nn.h"
#include "image.h"
#include "matrix.h"
#include "utils.h"

int main()
{
    NN nn(
        {
            {4, 3, "ReLU"}, 
            {3, 2, "Linear"}
        }, 
        0.3,
        true
    );

    Matrix input(4, 1);
    input.values = {{-1, 1, 3, 4}, {1, -1, 5, 6}};

    MatrixPtr a = nn.Forward(input);
    a->Print();

    std::cout << Accuracy(a->values, {{1}, {0}}) << "\n";
    std::cout << CrossEntropyLoss(a->values, {{1}, {0}}, 2);
    return 0;
}