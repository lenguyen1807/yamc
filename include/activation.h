#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>
#include <matrix.h>

class Activation
{
public:
    virtual Matrix Apply(const Matrix& mat, bool derivative = false)
    {
        Matrix res(mat);
        for (size_t i = 0; i < mat.rows; i++)
        {
            for (size_t j = 0; j < mat.cols; j++)
            {
                if (derivative)
                    res.values[i][j] = ComputeDer(mat.values[i][j]);
                else
                    res.values[i][j] = Compute(mat.values[i][j]);
            }
        }
        return res;
    }

    static Matrix SoftMax(const Matrix& mat)
    {
        // assume data is column vector
        assert(mat.cols == 1); 

        Matrix res(mat.rows, 1);

        // calculate sum_{i=0}^{n-1} e^{x_i}
        float sum = 0.0f;
        for (size_t i = 0; i < mat.rows; i++)
        {
            sum += exp(mat.values[i][0]);
        }

        for (size_t i = 0; i < mat.rows; i++)
        {
            res.values[i][0] = (exp(mat.values[i][0]) / sum);
        }

        return res;
    }

protected:
    virtual double Compute(double value) { return 0.0; }
    virtual double ComputeDer(double value) { return 0.0; }
};

class Sigmoid : public Activation
{
private:
    double Compute(double value)
    {
        return 1 / (1 + exp(-value));
    }

    double ComputeDer(double value)
    {
        return exp(-value) / pow((exp(-value) + 1), 2);
    }
};

class ReLU : public Activation
{
private:
    double Compute(double value)
    {
        // max(value, 0)
        return value ? value >= 0 : 0;
    }

    double ComputeDer(double value)
    {
        return 0 ? value <= 0 : 1;
    }
};

#endif // ACTIVATION_H