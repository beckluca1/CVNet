namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using DenseMatrixD = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix;

public static class CVHomography
{
    const double EPS = 1e-12;

    public static List<VectorD> HartleyNormalize(List<VectorD> points, out MatrixD T)
    {
        double cx = points.Average(p => p[0]);
        double cy = points.Average(p => p[1]);

        double meanDist = points
            .Average(p =>
            {
                double dx = p[0] - cx;
                double dy = p[1] - cy;
                return Math.Sqrt(dx * dx + dy * dy);
            });

        if (meanDist < EPS)
            throw new Exception($"Hartley Normalize mean Distance is {meanDist} so all points are to close together");

        double s = Math.Sqrt(2) / meanDist;

        T = DenseMatrixD.OfArray(new double[,]
        {
            { s, 0, -s * cx },
            { 0, s, -s * cy },
            { 0, 0, 1 }
        });

        var norm = new List<VectorD>();
        foreach (var p in points)
            norm.Add(DenseVectorD.OfArray([s * (p[0] - cx), s * (p[1] - cy), 1.0]));

        return norm;
    }

    public static MatrixD HartleyDenormalize(
        MatrixD H,
        MatrixD T1,
        MatrixD T2)
    {
        return T2.Inverse() * H * T1;
    }

    public static MatrixD HomographyNormalize(
    MatrixD H)
    {
        double scale = H[2, 2];

        if (Math.Abs(scale) < EPS)
            scale = H.FrobeniusNorm();

        if (Math.Abs(scale) < EPS)
            throw new Exception($"Homography Normalize scale factor is {scale} so neither last element nor Frobenius Norm yield good normalization");

        return H / scale;
    }

    private static MatrixD computeHomographyExact(List<VectorD> points1, List<VectorD> points2)
    {
        if (points1.Count != 4 || points2.Count != 4)
            throw new Exception($"Compute Homography Exact number of points are {points1.Count} / {points2.Count} but need exactly 4 points");

        var A = MatrixD.Build.Dense(8, 8);
        var b = VectorD.Build.Dense(8);

        for (int i = 0; i < 4; i++)
        {
            double x1 = points1[i][0];
            double y1 = points1[i][1];
            double x2 = points2[i][0];
            double y2 = points2[i][1];

            int r = i * 2;

            // X equation
            A[r + 0, 0] = x1;
            A[r + 0, 1] = y1;
            A[r + 0, 2] = 1;
            A[r + 0, 3] = 0;
            A[r + 0, 4] = 0;
            A[r + 0, 5] = 0;
            A[r + 0, 6] = -x2 * x1;
            A[r + 0, 7] = -x2 * y1;

            // Y equation
            A[r + 1, 0] = 0;
            A[r + 1, 1] = 0;
            A[r + 1, 2] = 0;
            A[r + 1, 3] = x1;
            A[r + 1, 4] = y1;
            A[r + 1, 5] = 1;
            A[r + 1, 6] = -y2 * x1;
            A[r + 1, 7] = -y2 * y1;

            b[r + 0] = x2;
            b[r + 1] = y2;
        }

        VectorD h = A.Solve(b);

        return MatrixD.Build.DenseOfArray(new double[,]
        {
            { h[0], h[1], h[2] },
            { h[3], h[4], h[5] },
            { h[6], h[7], 1.0  }
        });
    }

    private static MatrixD computeHomography(List<VectorD> points1, List<VectorD> points2)
    {
        DenseMatrixD A = DenseMatrixD.Create(2 * points1.Count, 9, 0);

        for (int i = 0; i < points1.Count; i++)
        {
            double x1 = points1[i][0];
            double y1 = points1[i][1];
            double x2 = points2[i][0];
            double y2 = points2[i][1];

            int r = i * 2;

            A[r + 0, 0] = -x1;
            A[r + 0, 1] = -y1;
            A[r + 0, 2] = -1;
            A[r + 0, 6] = x2 * x1;
            A[r + 0, 7] = x2 * y1;
            A[r + 0, 8] = x2;

            A[r + 1, 3] = -x1;
            A[r + 1, 4] = -y1;
            A[r + 1, 5] = -1;
            A[r + 1, 6] = y2 * x1;
            A[r + 1, 7] = y2 * y1;
            A[r + 1, 8] = y2;
        }

        var svd = A.Svd(true);

        if (svd.Rank < 8)
            throw new Exception($"Compute Homography insufficient Rank is {svd.Rank} but has to be 8 or more");

        VectorD h = svd.VT.Row(svd.VT.RowCount - 1);

        return DenseMatrixD.OfArray(new double[,]
        {
            { h[0], h[1], h[2] },
            { h[3], h[4], h[5] },
            { h[6], h[7], h[8] }
        });
    }

    public static MatrixD ComputeHomographyQuadExact(List<VectorD> points1, double targetWidth, double targetHeight)
    {
        List<VectorD> points2 =
        [
            DenseVectorD.OfArray([0.0, 0.0]),
            DenseVectorD.OfArray([targetWidth - 1, 0.0]),
            DenseVectorD.OfArray([targetWidth - 1, targetHeight - 1]),
            DenseVectorD.OfArray([0.0, targetHeight - 1]),
        ];

        return ComputeHomographyExact(points1, points2);
    }


    public static MatrixD ComputeHomographyExact(List<VectorD> points1, List<VectorD> points2)
    {
        if (points1.Count != 4 || points2.Count != 4)
            throw new Exception($"Compute Homography Exact number of points are {points1.Count} / {points2.Count} but need exactly 4 points");

        List<VectorD> normalizedPoints1 = HartleyNormalize(points1, out var T1);
        List<VectorD> normalizedPoints2 = HartleyNormalize(points2, out var T2);

        MatrixD homography = computeHomographyExact(normalizedPoints1, normalizedPoints2);
        homography = HartleyDenormalize(homography, T1, T2);
        homography = HomographyNormalize(homography);

        return homography;
    }

    public static MatrixD ComputeHomography(List<VectorD> points1, List<VectorD> points2)
    {
        if (points1.Count != points2.Count)
            throw new Exception($"Compute Homography Number of points are {points1.Count} / {points2.Count} but they have to be the same amount");

        List<VectorD> normalizedPoints1 = HartleyNormalize(points1, out var T1);
        List<VectorD> normalizedPoints2 = HartleyNormalize(points2, out var T2);

        MatrixD homography = computeHomography(normalizedPoints1, normalizedPoints2);

        homography = HartleyDenormalize(homography, T1, T2);
        homography = HomographyNormalize(homography);

        return homography;
    }

    public static MatrixD ComputeHomographyRansac(
        List<VectorD> points1,
        List<VectorD> points2,
        int iterations,
        double threshold,
        out List<int> bestInliers,
        out double bestError)
    {
        bestInliers = new List<int>();
        bestError = double.MaxValue;

        if (points1.Count != points2.Count)
            throw new Exception($"Compute Homography Number of points are {points1.Count} / {points2.Count} but they have to be the same amount");

        List<VectorD> normalizedPoints1 = HartleyNormalize(points1, out var T1);
        List<VectorD> normalizedPoints2 = HartleyNormalize(points2, out var T2);

        int n = points1.Count;
        for (int iter = 0; iter < iterations; iter++)
        {
            CVCamera.Ransac2Select(normalizedPoints1, normalizedPoints2, 4, out List<VectorD> sample1, out List<VectorD> sample2);

            try
            {
                MatrixD homography = computeHomographyExact(sample1, sample2);
                homography = HartleyDenormalize(homography, T1, T2);
                homography = HomographyNormalize(homography);

                MatrixD homographyInverse = homography.Inverse();

                List<VectorD> projectedPoints1 = CVProjection.HomographyProjectPoints(points1, homography);
                List<VectorD> unprojectedPoints2 = CVProjection.HomographyProjectPoints(points2, homographyInverse);

                // Count inliers
                List<int> inliers = new();
                double inlierError = 0.0;
                for (int i = 0; i < n; i++)
                {
                    double dx1 = projectedPoints1[i][0] - points2[i][0];
                    double dy1 = projectedPoints1[i][1] - points2[i][1];
                    double homographyError1 = Math.Sqrt(dx1 * dx1 + dy1 * dy1);

                    double dx2 = points1[i][0] - unprojectedPoints2[i][0];
                    double dy2 = points1[i][1] - unprojectedPoints2[i][1];
                    double homographyError2 = Math.Sqrt(dx2 * dx2 + dy2 * dy2);

                    double homographyError = (homographyError1 + homographyError2) / 2;

                    if (homographyError < threshold)
                    {
                        inlierError += homographyError;
                        inliers.Add(i);
                    }
                }

                if (inliers.Count > 0)
                    inlierError /= inliers.Count;
                else
                    inlierError = double.MaxValue;

                // Keep best model
                if (inliers.Count > bestInliers.Count || (inliers.Count == bestInliers.Count && inlierError < bestError))
                {
                    bestInliers = inliers;
                    bestError = inlierError;
                }
            }
            catch { }
        }

        if (bestInliers.Count < 4)
            throw new Exception("No valid homography found");

        // Recalculate using all inliers
        List<VectorD> final1 = new();
        List<VectorD> final2 = new();

        foreach (int i in bestInliers)
        {
            final1.Add(points1[i]);
            final2.Add(points2[i]);
        }

        return ComputeHomography(final1, final2);
    }
};