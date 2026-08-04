using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace CVNet;

public class CVCalibration
{
    public static double Calibrate(List<List<Vector<double>>> imagePointGroups, List<List<Vector<double>>> worldPointGroups, out Matrix<double> K, out Vector<double> d)
    {
        if (imagePointGroups.Count == 0 || imagePointGroups.Count != worldPointGroups.Count) throw new Exception("Invalid number of samples");

        List<Matrix<double>> homographies = new List<Matrix<double>>();

        for (int i = 0; i < imagePointGroups.Count; i++)
        {
            if (imagePointGroups[i].Count == 0 || imagePointGroups[i].Count != worldPointGroups[i].Count) throw new Exception("Invalid number of points");

            homographies.Add(CVCamera.ComputeHomography(imagePointGroups[i], worldPointGroups[i]));
        }

        K = CVCamera.ComputeIntrinsics(homographies);

        CVCamera.ComputeExtrinsics(homographies, K, out List<Matrix<double>> Rs, out List<Vector<double>> ts);

        List<List<Vector<double>>> transformedWrldPoints = CVCamera.TransformPoints(worldPointGroups, Rs, ts);

        List<Vector<double>> imagePoints = new List<Vector<double>>();
        List<Vector<double>> worldPoints = new List<Vector<double>>();

        for (int i = 0; i < imagePointGroups.Count; i++)
        {
            foreach (Vector<double> imagePoint in imagePointGroups[i])
                imagePoints.Add(imagePoint);
            foreach (Vector<double> worldPoint in transformedWrldPoints[i])
                worldPoints.Add(worldPoint);
        }

        Vector<double> distortion = DenseVector.OfArray([0, 0, 0, 0, 0]);

        Console.WriteLine(CVCamera.ComputeError(worldPoints, imagePoints, K, distortion));

        d = CVCamera.LevenbergMarquardt(worldPoints, imagePoints, K, distortion);

        return CVCamera.ComputeError(worldPoints, imagePoints, K, d);
    }
};