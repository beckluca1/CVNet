using System.Numerics;

namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using DenseMatrixD = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix;

public static class CVResizeTransformation
{
    // Optimized
    public static VectorD Stretch(
        int currentWidth,
        int currentHeight,
        VectorD pointIn,
        int targetWidth,
        int targetHeight)
    {
        if (pointIn.Count == 2)
            return DenseVectorD.OfArray([pointIn[0] * targetWidth / currentWidth, pointIn[1] * targetHeight / currentHeight]);
        else if (pointIn.Count == 3)
            return DenseVectorD.OfArray([pointIn[0] * targetWidth / currentWidth, pointIn[1] * targetHeight / currentHeight, pointIn[2]]);
        else
            throw new ArgumentException($"Unsupported vector size {pointIn.Count}");
    }

    // Optimized
    public static VectorD CopyCenter(
        int currentWidth,
        int currentHeight,
        VectorD pointIn,
        int targetWidth,
        int targetHeight)
    {
        double dx = (currentWidth - targetWidth) / 2.0;
        double dy = (currentHeight - targetHeight) / 2.0;

        if (pointIn.Count == 2)
            return DenseVectorD.OfArray([pointIn[0] - dx, pointIn[1] - dy]);
        else if (pointIn.Count == 3)
            return DenseVectorD.OfArray([pointIn[0] - dx, pointIn[1] - dy, pointIn[2]]);
        else
            throw new ArgumentException($"Unsupported vector size {pointIn.Count}");
    }
}