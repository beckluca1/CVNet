namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;

public class CVProjection
{
    public static VectorD HomographyProjectPoint(VectorD point, MatrixD homography)
    {
        var p = DenseVectorD.OfArray([point[0], point[1], 1.0]);
        var r = homography * p;

        if (Math.Abs(r[2]) < 1e-12)
            return DenseVectorD.OfArray([double.NaN, double.NaN, double.NaN]);

        double x = r[0] / r[2];
        double y = r[1] / r[2];

        return DenseVectorD.OfArray([x, y, 1.0]);
    }

    public static List<VectorD> HomographyProjectPoints(List<VectorD> points, MatrixD homography)
    {
        List<VectorD> projectedPoints = new List<VectorD>();

        foreach (VectorD point in points)
        {
            projectedPoints.Add(HomographyProjectPoint(point, homography));
        }

        return projectedPoints;
    }

    public static List<List<VectorD>> HomographyProjectPoints(List<List<VectorD>> points, MatrixD homography)
    {
        List<List<VectorD>> projectedPoints = new List<List<VectorD>>();

        for (int i = 0; i < points.Count; i++)
        {
            projectedPoints.Add(HomographyProjectPoints(points[i], homography));
        }

        return projectedPoints;
    }

    public static VectorD IntrinsicProjectPoint(
        VectorD point,
        MatrixD K,
        VectorD distortion)
    {
        if (Math.Abs(point[2]) < 1e-12)
            return DenseVectorD.OfArray([double.NaN, double.NaN, double.NaN]);

        double fx = K[0, 0];
        double fy = K[1, 1];
        double cx = K[0, 2];
        double cy = K[1, 2];

        double x = point[0] / point[2];
        double y = point[1] / point[2];

        double r2 = x * x + y * y;
        double r4 = r2 * r2;
        double r6 = r4 * r2;

        double k1 = distortion[0];
        double k2 = distortion[1];
        double p1 = distortion[2];
        double p2 = distortion[3];
        double k3 = distortion[4];

        double radial = 1 + k1 * r2 + k2 * r4 + k3 * r6;

        double xDist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
        double yDist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;

        return DenseVectorD.OfArray([fx * xDist + cx, fy * yDist + cy, 1]);
    }

    public static List<VectorD> IntrinsicProjectPoints(List<VectorD> points, MatrixD K, VectorD distortion)
    {
        List<VectorD> projectedPoints = new List<VectorD>();

        foreach (VectorD point in points)
        {
            projectedPoints.Add(IntrinsicProjectPoint(point, K, distortion));
        }

        return projectedPoints;
    }

    public static List<List<VectorD>> IntrinsicProjectPoints(List<List<VectorD>> points, MatrixD K, VectorD distortion)
    {
        List<List<VectorD>> projectedPoints = new List<List<VectorD>>();

        for (int i = 0; i < points.Count; i++)
        {
            projectedPoints.Add(IntrinsicProjectPoints(points[i], K, distortion));
        }

        return projectedPoints;
    }

    public static VectorD IntrinsicUnProjectPoint(VectorD pixel, MatrixD K, VectorD distortion)
    {
        double fx = K[0, 0], fy = K[1, 1];
        double cx = K[0, 2], cy = K[1, 2];

        double k1 = distortion[0];
        double k2 = distortion[1];
        double p1 = distortion[2];
        double p2 = distortion[3];
        double k3 = distortion[4];

        // distorted normalized coordinates
        double xd = (pixel[0] - cx) / fx;
        double yd = (pixel[1] - cy) / fy;

        // initial guess
        double x = xd;
        double y = yd;

        for (int i = 0; i < 5; i++)
        {
            double r2 = x * x + y * y;
            double r4 = r2 * r2;
            double r6 = r4 * r2;

            double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

            double deltaX = 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
            double deltaY = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;

            x = (xd - deltaX) / radial;
            y = (yd - deltaY) / radial;
        }

        return DenseVectorD.OfArray([x, y, 1]);
    }

    public static List<VectorD> IntrinsicUnProjectPoints(List<VectorD> points, MatrixD K, VectorD distortion)
    {
        List<VectorD> projectedPoints = new List<VectorD>();

        foreach (VectorD point in points)
        {
            projectedPoints.Add(IntrinsicUnProjectPoint(point, K, distortion));
        }

        return projectedPoints;
    }

    public static VectorD ExtrinsicProjectPoint(VectorD point, MatrixD R, VectorD t)
    {
        VectorD pC = R * point + t;
        return pC / pC[2];
    }

    public static List<VectorD> ExtrinsicProjectPoints(List<VectorD> points, MatrixD R, VectorD t)
    {
        List<VectorD> transformedPoints = new List<VectorD>();

        foreach (VectorD point in points)
        {
            transformedPoints.Add(ExtrinsicProjectPoint(point, R, t));
        }

        return transformedPoints;
    }

    public static VectorD ProjectPoint(VectorD point, MatrixD R, VectorD t, MatrixD K, VectorD distortion)
    {
        VectorD pC = ExtrinsicProjectPoint(point, R, t);
        return IntrinsicProjectPoint(pC, K, distortion);
    }

    public static List<VectorD> ProjectPoints(List<VectorD> points, MatrixD R, VectorD t, MatrixD K, VectorD distortion)
    {
        List<VectorD> transformedPoints = new List<VectorD>();

        foreach (VectorD point in points)
        {
            transformedPoints.Add(ProjectPoint(point, R, t, K, distortion));
        }

        return transformedPoints;
    }
};