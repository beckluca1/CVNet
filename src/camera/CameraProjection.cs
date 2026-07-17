using MathNet.Numerics.LinearAlgebra;

namespace CVNet;

public class CVProjection
{
    public static Vector<double> UndistortPoint(
        Vector<double> p,
        Matrix<double> K,
        Vector<double> d)
    {
        double fx = K[0, 0];
        double fy = K[1, 1];
        double cx = K[0, 2];
        double cy = K[1, 2];

        double k1 = d[0];
        double k2 = d[1];
        double p1 = d[2];
        double p2 = d[3];
        double k3 = d.Count > 4 ? d[4] : 0.0;

        // Step 1: pixel -> normalized distorted coords
        double x = (p[0] - cx) / fx;
        double y = (p[1] - cy) / fy;

        // Step 2: iterative undistortion (fixed-point)
        double x0 = x;
        double y0 = y;

        for (int i = 0; i < 5; i++)
        {
            double r2 = x0 * x0 + y0 * y0;
            double radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;

            double xTang = 2 * p1 * x0 * y0 + p2 * (r2 + 2 * x0 * x0);
            double yTang = p1 * (r2 + 2 * y0 * y0) + 2 * p2 * x0 * y0;

            double xEst = (x - xTang) / radial;
            double yEst = (y - yTang) / radial;

            x0 = xEst;
            y0 = yEst;
        }

        return Vector<double>.Build.DenseOfArray(new[] { x0, y0 });
    }

    public static List<Vector<double>> UndistortPoints(List<Vector<double>> imagePoints, Matrix<double> K, Vector<double> d)
    {
        List<Vector<double>> undistortedPoints = new List<Vector<double>>();

        foreach (Vector<double> imagePoint in imagePoints)
        {
            undistortedPoints.Add(UndistortPoint(imagePoint, K, d));
        }

        return undistortedPoints;
    }

    public static void CalculatePlaneProjection(List<Vector<double>> imagePoints, List<Vector<double>> worldPoints, Matrix<double> K, Vector<double> d, out Matrix<double> R, out Vector<double> t)
    {
        Matrix<double> homography = CVCamera.ComputeHomographyStable(imagePoints, worldPoints);

        Matrix<double> Kinv = K.Inverse();

        CVCamera.ComputeExtrinsics(homography, Kinv, out Matrix<double> RO, out Vector<double> tO);

        R = RO;
        t = tO;
    }
};