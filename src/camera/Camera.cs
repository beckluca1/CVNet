namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using DenseMatrixD = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix;

public static partial class CVCamera
{
    public static Random random = new Random();

    public static void Ransac2Select<T>(List<T> inputList1, List<T> inputList2, int samples, out List<T> items1, out List<T> items2)
    {
        int n = inputList1.Count;
        HashSet<int> indices = new HashSet<int>();

        while (indices.Count < samples)
            indices.Add(random.Next(n));

        items1 = new();
        items2 = new();

        foreach (int i in indices)
        {
            items1.Add(inputList1[i]);
            items2.Add(inputList2[i]);
        }
    }

    public static void EstimateCameraPoseRansac5Point(
        List<VectorD> camera1Points,
        List<VectorD> camera2Points,
        MatrixD K1,
        MatrixD K2,
        VectorD d1,
        VectorD d2,
        int iterations,
        double thresholdPixels,
        out MatrixD R,
        out VectorD t,
        out List<int> bestInliers)
    {
        R =
            DenseMatrixD.CreateIdentity(3);

        t =
            DenseVectorD.OfArray([0, 0, 0]);

        bestInliers =
            new List<int>();

        if (camera1Points.Count != camera2Points.Count)
            throw new ArgumentException(
                "Point lists must have same length.");

        if (camera1Points.Count < 5)
            throw new ArgumentException(
                "At least 5 correspondences are required.");

        // ------------------------------------------------------------
        // 1. Pixel -> calibrated normalized coordinates
        // ------------------------------------------------------------

        List<VectorD> p1 =
            CVProjection.IntrinsicUnProjectPoints(
                camera1Points,
                K1,
                d1);

        List<VectorD> p2 =
            CVProjection.IntrinsicUnProjectPoints(
                camera2Points,
                K2,
                d2);

        // ------------------------------------------------------------
        // 2. Pixel threshold -> normalized Sampson threshold
        // ------------------------------------------------------------

        double focal =
            0.5 * (K1[0, 0] + K1[1, 1]);

        double normalizedThreshold =
            thresholdPixels / focal;

        double sampsonThreshold =
            normalizedThreshold *
            normalizedThreshold;

        // ------------------------------------------------------------
        // 3. 5-point RANSAC
        // ------------------------------------------------------------

        bool success =
            CVHomographySolver.EstimateEssentialRansac(
                p1,
                p2,
                iterations,
                sampsonThreshold,
                0.999,
                out MatrixD E,
                out bestInliers);

        if (!success)
            throw new Exception(
                "5-point RANSAC failed.");

        if (bestInliers.Count < 5)
            throw new Exception(
                "Not enough inliers.");

        // ------------------------------------------------------------
        // 4. Refine E using all RANSAC inliers
        // ------------------------------------------------------------

        List<VectorD> inlierP1 = new();
        List<VectorD> inlierP2 = new();

        foreach (int i in bestInliers)
        {
            inlierP1.Add(p1[i]);
            inlierP2.Add(p2[i]);
        }

        E =
            NormalizedEstimateEssential(
                inlierP1,
                inlierP2);

        E =
            EnforceEssential(E);

        // ------------------------------------------------------------
        // 5. Decompose E
        // ------------------------------------------------------------

        DecomposeEssential(
            E,
            out List<MatrixD> rotations,
            out List<VectorD> translations);

        // ------------------------------------------------------------
        // 6. Select the correct R/t using cheirality
        // ------------------------------------------------------------

        NormalizedSelectPose(
            rotations,
            translations,
            inlierP1,
            inlierP2,
            inlierP1,
            inlierP2,
            K1, K2, d1, d2, out R, out t);

        // IMPORTANT:
        //
        // DO NOT independently flip t.
        //
        // R and t are a coupled solution.
    }

    public static List<VectorD> TriangulateAll(
        List<VectorD> src,
        List<VectorD> dst,
        MatrixD K1,
        MatrixD K2,
        VectorD d1,
        VectorD d2,
        MatrixD R,
        VectorD t)
    {
        if (src.Count != dst.Count)
            throw new ArgumentException("Point lists must have same length");

        List<VectorD> normalizedPixels1 = CVProjection.IntrinsicUnProjectPoints(src, K1, d1);
        List<VectorD> normalizedPixels2 = CVProjection.IntrinsicUnProjectPoints(dst, K2, d2);

        var points = new List<VectorD>();

        for (int i = 0; i < normalizedPixels1.Count; i++)
        {
            VectorD X = NormalizedTriangulate(R, t, normalizedPixels1[i], normalizedPixels2[i]);
            points.Add(X);
        }

        return points;
    }

    public static void StereoRectify(
        MatrixD K1,
        MatrixD K2,
        MatrixD R,
        VectorD t,
        out MatrixD R1,
        out MatrixD R2,
        out MatrixD H1,
        out MatrixD H2)
    {
        VectorD C1 = DenseVectorD.OfArray([0, 0, 0]);
        VectorD C2 = -R.Transpose() * t;

        // For stereo rigs and camera2 is always in the same direction relative to camera1
        if (C2[0] < 0)
            C2 = R.Transpose() * t;

        VectorD ex = (C2 - C1).Normalize(2);

        VectorD z1 = DenseVectorD.OfArray([0, 0, 1]);
        VectorD z2 = R.Transpose() * z1;

        VectorD ez = (z1 + z2).Normalize(2);

        VectorD ey = Cross(ez, ex).Normalize(2);
        ez = Cross(ex, ey).Normalize(2);

        MatrixD Rrect = DenseMatrixD.OfRowVectors(ex, ey, ez);

        if (Rrect.Determinant() < 0)
        {
            ez = -ez;
            Rrect = DenseMatrixD.OfRowVectors(ex, ey, ez);
        }

        R1 = Rrect;
        R2 = Rrect * R.Transpose();

        H1 = K1 * R1 * K1.Inverse();
        H2 = K2 * R2 * K2.Inverse();

        if (H1[2, 2] != 0.0)
            H1 /= H1[2, 2];
        if (H2[2, 2] != 0.0)
            H2 /= H2[2, 2];
    }

    public static void StereoRectifyRansac(
            List<VectorD> src,
            List<VectorD> dst,
            MatrixD K1,
            MatrixD K2,
            VectorD d1,
            VectorD d2,
            int iterations,
            double threshold,
            out MatrixD R,
            out VectorD t,
            out MatrixD R1,
            out MatrixD R2,
            out MatrixD H1,
            out MatrixD H2)
    {
        EstimateCameraPoseRansac(src, dst, K1, K2, d1, d2, iterations, threshold, out _, out R, out t, out List<int> inliers);
        StereoRectify(K1, K2, R, t, out R1, out R2, out H1, out H2);
    }

    public static void CalculatePlaneProjection(List<VectorD> imagePoints, List<VectorD> worldPoints, MatrixD K, VectorD d, out MatrixD R, out VectorD t)
    {
        MatrixD homography = CVHomography.ComputeHomography(imagePoints, worldPoints);

        MatrixD Kinv = K.Inverse();

        CVCalibration.ComputeExtrinsics(homography, Kinv, out MatrixD RO, out VectorD tO);

        R = RO;
        t = tO;
    }
};