using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace CVNet;

public class CVCheckerboard
{
    public static List<Vector<double>> DetectCheckerboard(CVImage image)
    {
        List<Vector<double>> saddlePoints = new List<Vector<double>>();

        var corners = CVCornerDetector.DetectCornerHessian(image, 2, 5);
        for (int i = 0; i < corners.Count; i++) saddlePoints.Add(DenseVector.OfArray([corners[i].Item1, corners[i].Item2, corners[i].Item3]));

        Console.WriteLine($"Saddle points: {saddlePoints.Count}");

        return saddlePoints;
    }
}