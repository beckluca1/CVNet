#if DEBUG
namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using DenseMatrixD = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix;

using Xunit;

public class CameraPoseTests
{
    private const double _pixelNoise = 10.0;
    private const double _badMatches = 0.1;

    [Fact]
    public void TestEstimateCameraPoseRansac()
    {
        Random random = new Random();

        MatrixD R1 = DenseMatrixD.CreateIdentity(3);
        MatrixD R2 = DenseMatrixD.CreateIdentity(3);

        VectorD t1 = DenseVectorD.OfArray([0, 0, -10.0f]);
        VectorD t2 = DenseVectorD.OfArray([-5.0f, 0, -10.0f]);

        MatrixD K = DenseMatrixD.OfArray(new double[,] { { 500, 0, 250 }, { 0, 500, 250 }, { 0, 0, 1 } });
        VectorD d = DenseVectorD.Build.Dense(5);

        List<VectorD> projectedPoints1 = new List<VectorD>();
        List<VectorD> projectedPoints2 = new List<VectorD>();
        for (int i = 0; i < 100; i++)
        {
            VectorD worldPoint1 = DenseVectorD.OfArray([random.NextDouble() * 10.0, random.NextDouble() * 10.0, random.NextDouble() * 0.1]);
            VectorD worldPoint2 = worldPoint1;
            VectorD noise1 = DenseVectorD.OfArray([random.NextDouble() * _pixelNoise, random.NextDouble() * _pixelNoise, random.NextDouble() * _pixelNoise]);
            VectorD noise2 = DenseVectorD.OfArray([random.NextDouble() * _pixelNoise, random.NextDouble() * _pixelNoise, random.NextDouble() * _pixelNoise]);

            // Bad match
            if (random.NextDouble() > 1.0 - _badMatches) worldPoint2 = DenseVectorD.OfArray([random.NextDouble() * 10.0, random.NextDouble() * 10.0, 0.0]);

            projectedPoints1.Add(CVProjection.ProjectPoint(worldPoint1, R1, t1, K, d) + noise1);
            projectedPoints2.Add(CVProjection.ProjectPoint(worldPoint2, R2, t2, K, d) + noise2);
        }

        CVCamera.EstimateCameraPoseRansac(projectedPoints1, projectedPoints2, K, K, d, d, 2000, 10e-4, out _, out MatrixD R, out VectorD t, out _);
        Console.WriteLine(R);
        Console.WriteLine(t);
    }
}
#endif