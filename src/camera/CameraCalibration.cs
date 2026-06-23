using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace CVNet;

public class CVCalibration
{
    public static double Calibrate(List<List<Vector<double>>> imagePointGroups, List<List<Vector<double>>> worldPointGroup, out Matrix<double> K, out Vector<double> d)
    {
        List<Matrix<double>> homographies = new List<Matrix<double>>();

        for (int i = 0; i < imagePointGroups.Count; i++)
        {
            List<Vector<double>> normalizedImagePoints = CVCamera.Normalize(imagePointGroups[i], out var TImg);
            List<Vector<double>> normalizedWorldPoints = CVCamera.Normalize(worldPointGroup[i], out var TWrld);

            Matrix<double> homography = CVCamera.ComputeHomography(normalizedImagePoints, normalizedWorldPoints);

            homographies.Add(CVCamera.Denormalize(homography, TImg, TWrld));
        }

        K = CVCamera.ComputeIntrinsics(homographies);

        CVCamera.ComputeExtrinsics(homographies, K, out List<Matrix<double>> Rs, out List<Vector<double>> ts);

        List<List<Vector<double>>> transformedWrldPoints = CVCamera.TransformPoints(worldPointGroup, Rs, ts);

        List<Vector<double>> imagePoints = new List<Vector<double>>();
        List<Vector<double>> worldPoints = new List<Vector<double>>();

        for (int i = 0; i < imagePointGroups.Count; i++)
        {
            foreach (Vector<double> imagePoint in imagePointGroups[i])
                imagePoints.Add(imagePoint);
            foreach (Vector<double> worldPoint in transformedWrldPoints[i])
                worldPoints.Add(worldPoint);
        }

        Vector<double> dist = DenseVector.OfArray(new double[] { 0, 0, 0, 0, 0 });

        d = CVCamera.LevenbergMarquardt(worldPoints, imagePoints, K, dist);

        return CVCamera.ComputeError(worldPoints, imagePoints, K, d);
    }
};