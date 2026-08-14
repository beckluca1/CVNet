namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using DenseMatrixD = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix;

public static class CVCamera
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

    public static VectorD Cross(
            VectorD a,
            VectorD b)
    {
        return DenseVectorD.OfArray(
        [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ]);
    }

    private static MatrixD EnforceEssential(
        MatrixD E)
    {
        var svd = E.Svd(true);

        var U = svd.U;
        var Vt = svd.VT;

        // Ensure proper rotations
        if (U.Determinant() < 0)
            U.SetColumn(2, -U.Column(2));

        if (Vt.Transpose().Determinant() < 0)
            Vt.SetRow(2, -Vt.Row(2));

        double s = (svd.S[0] + svd.S[1]) * 0.5;

        var S = DenseMatrixD.OfArray(new double[,] {
            {s,0,0},
            {0,s,0},
            {0,0,0}
        });

        return U * S * Vt;
    }


    private static MatrixD EstimateEssential(
        List<VectorD> p1,
        List<VectorD> p2)
    {
        int n = p1.Count;

        if (n < 8) throw new Exception("Essential matrix requires at least 8 points");

        var A = DenseMatrixD.Create(n, 9, 0);

        for (int i = 0; i < n; i++)
        {
            double x1 = p1[i][0];
            double y1 = p1[i][1];

            double x2 = p2[i][0];
            double y2 = p2[i][1];

            A[i, 0] = x2 * x1;
            A[i, 1] = x2 * y1;
            A[i, 2] = x2;

            A[i, 3] = y2 * x1;
            A[i, 4] = y2 * y1;
            A[i, 5] = y2;

            A[i, 6] = x1;
            A[i, 7] = y1;
            A[i, 8] = 1;
        }

        var svd = A.Svd(true);
        VectorD e = svd.VT.Row(8);

        var E = DenseMatrixD.OfArray(new double[,] {
            {e[0],e[1],e[2]},
            {e[3],e[4],e[5]},
            {e[6],e[7],e[8]}
        });

        return E;
    }

    private static void DecomposeEssential(
       MatrixD E,
       out List<MatrixD> rotations,
       out List<VectorD> translations)
    {
        var svd = E.Svd(true);

        var U = svd.U;
        var Vt = svd.VT;

        // Fix SVD signs
        if (U.Determinant() < 0)
            U.SetColumn(2, -U.Column(2));

        if (Vt.Transpose().Determinant() < 0)
            Vt.SetRow(2, -Vt.Row(2));

        var W = DenseMatrixD.OfArray(new double[,] {
            {0,-1,0},
            {1, 0,0},
            {0, 0,1}
        });

        MatrixD R1 = U * W * Vt;
        MatrixD R2 = U * W.Transpose() * Vt;

        VectorD t = U.Column(2);

        rotations = new List<MatrixD> { R1, R1, R2, R2 };
        translations = new List<VectorD> { t, -t, t, -t };
    }

    private static VectorD triangulate(
        MatrixD R,
        VectorD t,
        VectorD x1,
        VectorD x2)
    {
        //Camera 1: P1 = [I | 0]
        //Camera 2: P2 = [R | t]

        var P1 = DenseMatrixD.OfArray(new double[,]
        {
            {1,0,0,0},
            {0,1,0,0},
            {0,0,1,0}
        });

        var P2 = DenseMatrixD.Create(3, 4, 0);
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
                P2[r, c] = R[r, c];

            P2[r, 3] = t[r];
        }

        var A = DenseMatrixD.Create(4, 4, 0);

        // first camera
        A.SetRow(0, x1[0] * P1.Row(2) - P1.Row(0));
        A.SetRow(1, x1[1] * P1.Row(2) - P1.Row(1));

        // second camera
        A.SetRow(2, x2[0] * P2.Row(2) - P2.Row(0));
        A.SetRow(3, x2[1] * P2.Row(2) - P2.Row(1));

        var svd = A.Svd(true);

        VectorD X = svd.VT.Row(3);

        if (Math.Abs(X[3]) < 1e-12)
            return DenseVectorD.OfArray([double.NaN, double.NaN, double.NaN]);

        return DenseVectorD.OfArray([X[0] / X[3], X[1] / X[3], X[2] / X[3]]);
    }

    private static int SelectPose(
        List<MatrixD> rotations,
        List<VectorD> translations,
        List<VectorD> p1,
        List<VectorD> p2)
    {
        int bestIndex = -1;
        int bestCount = -1;
        double bestError = double.MaxValue;

        for (int pose = 0; pose < rotations.Count; pose++)
        {
            int count = 0;
            double error = 0.0;

            for (int i = 0; i < p1.Count; i++)
            {
                VectorD X1 = triangulate(rotations[pose], translations[pose], p1[i], p2[i]);
                VectorD X2 = rotations[pose] * X1 + translations[pose];

                if (double.IsNaN(X1[0]))
                    continue;

                double z1 = X1[2];
                double z2 = X2[2];

                if (z1 > 0 && z2 > 0)
                {
                    count++;

                    VectorD x1 = X1 / X1[2];
                    VectorD x2 = rotations[pose] * X1 + translations[pose];
                    x2 /= x2[2];

                    error +=
                        Math.Pow(x1[0] - p1[i][0], 2) +
                        Math.Pow(x1[1] - p1[i][1], 2) +
                        Math.Pow(x2[0] - p2[i][0], 2) +
                        Math.Pow(x2[1] - p2[i][1], 2);
                }
            }

            if (count > bestCount || (count == bestCount && error < bestError))
            {
                bestCount = count;
                bestIndex = pose;
                bestError = error;
            }
        }

        if (bestIndex < 0)
            throw new Exception("No valid camera pose found");

        return bestIndex;
    }

    private static void estimateCameraPose(
           List<VectorD> srcHNorm,
           List<VectorD> dstHNorm,
           MatrixD T1,
           MatrixD T2,
           out MatrixD E,
           out List<MatrixD> Rs,
           out List<VectorD> ts)
    {
        MatrixD En = EstimateEssential(srcHNorm, dstHNorm);

        // Undo Hartley normalization
        E = CVHomography.HartleyDenormalize(En, T1, T2);
        E = EnforceEssential(E);

        // Four possible poses
        DecomposeEssential(E, out Rs, out ts);
    }

    public static double EstimateCameraPose(
        List<VectorD> src,
        List<VectorD> dst,
        MatrixD K1,
        MatrixD K2,
        VectorD d1,
        VectorD d2,
        out MatrixD R,
        out VectorD t)
    {
        List<VectorD> srcNorm = CVProjection.IntrinsicUnProjectPoints(src, K1, d1);
        List<VectorD> dstNorm = CVProjection.IntrinsicUnProjectPoints(dst, K2, d2);

        // Estimate essential matrix
        List<VectorD> points1 = CVHomography.HartleyNormalize(srcNorm, out MatrixD T1);
        List<VectorD> points2 = CVHomography.HartleyNormalize(dstNorm, out MatrixD T2);

        estimateCameraPose(points1, points2, T1, T2, out MatrixD E, out var Rs, out var ts);

        // Pick the pose where points are in front
        // of both cameras
        int index = SelectPose(Rs, ts, srcNorm, dstNorm);

        R = Rs[index];
        t = ts[index];

        // Flip direction so its consistent
        if ((R.Transpose() * t)[0] < 0)
            t = -t;

        // double meanReprojectionError = MeanReprojectionError(src, dst, R, t, K1, K2, d1, d2);
        // Console.WriteLine($"Mean reprojection error: {meanReprojectionError}");

        return MeanSampsonError(srcNorm, dstNorm, E);
    }

    public static double MeanReprojectionError(List<VectorD> src,
                                                List<VectorD> dst,
                                                MatrixD R,
                                                VectorD t,
                                                MatrixD K1,
                                                MatrixD K2,
                                                VectorD d1,
                                                VectorD d2)
    {
        List<VectorD> normalizedPixels1 = CVProjection.IntrinsicUnProjectPoints(src, K1, d1);
        List<VectorD> normalizedPixels2 = CVProjection.IntrinsicUnProjectPoints(dst, K2, d2);

        double error = 0.0;
        double errorCam1 = 0.0;
        double errorCam2 = 0.0;

        for (int i = 0; i < src.Count; i++)
        {
            VectorD worldPoint = triangulate(R, t, normalizedPixels1[i], normalizedPixels2[i]);

            VectorD p1 = CVProjection.ProjectPoint(worldPoint, DenseMatrixD.CreateIdentity(3), DenseVectorD.OfArray([0, 0, 0]), K1, d1);
            VectorD p2 = CVProjection.ProjectPoint(worldPoint, R, t, K2, d2);

            // 4. Compute errors
            double dx1 = p1[0] - src[i][0];
            double dy1 = p1[1] - src[i][1];
            double dx2 = p2[0] - dst[i][0];
            double dy2 = p2[1] - dst[i][1];
            errorCam1 += dx1 * dx1 + dy1 * dy1;
            errorCam2 += dx2 * dx2 + dy2 * dy2;
            error += dx1 * dx1 + dy1 * dy1 + dx2 * dx2 + dy2 * dy2;
        }

        // Console.WriteLine($"Cam1 RMS {Math.Sqrt(errorCam1 / (src.Count * 4))}, Cam2 RMS {Math.Sqrt(errorCam2 / (src.Count * 4))}");

        return Math.Sqrt(error / (src.Count * 4));
    }

    public static double MeanSampsonError(
            List<VectorD> srcNorm,
            List<VectorD> dstNorm,
            MatrixD E)
    {
        double error = 0.0;

        MatrixD Et = E.Transpose();

        for (int i = 0; i < srcNorm.Count; i++)
        {
            VectorD Ex1 = E * srcNorm[i];
            VectorD Etx2 = Et * dstNorm[i];

            double numer = Math.Pow(dstNorm[i].DotProduct(Ex1), 2);

            double denom =
                Ex1[0] * Ex1[0] +
                Ex1[1] * Ex1[1] +
                Etx2[0] * Etx2[0] +
                Etx2[1] * Etx2[1];

            if (denom > 1e-12)
                error += numer / denom;
        }

        return error / srcNorm.Count;
    }

    public static double MeanSampsonErrorInliers(
        List<VectorD> srcNorm,
        List<VectorD> dstNorm,
        MatrixD E,
        double threshold,
        out List<int> inliers)
    {
        inliers = new();
        double error = 0.0;

        MatrixD Et = E.Transpose();

        for (int i = 0; i < srcNorm.Count; i++)
        {
            VectorD Ex1 = E * srcNorm[i];
            VectorD Etx2 = Et * dstNorm[i];

            double numer = Math.Pow(dstNorm[i].DotProduct(Ex1), 2);

            double denom =
                Ex1[0] * Ex1[0] +
                Ex1[1] * Ex1[1] +
                Etx2[0] * Etx2[0] +
                Etx2[1] * Etx2[1];

            double sampsonError = numer / denom;

            if (sampsonError < threshold)
            {
                error += sampsonError;
                inliers.Add(i);
            }
        }

        return error / inliers.Count;
    }

    public static void EstimateCameraPoseRansac(
            List<VectorD> camera1Points,
            List<VectorD> camera2Points,
            MatrixD K1,
            MatrixD K2,
            VectorD d1,
            VectorD d2,
            int iterations,
            double threshold,
            out MatrixD R,
            out VectorD t,
            out List<int> bestInliers)
    {
        R = DenseMatrixD.CreateIdentity(3);
        t = DenseVectorD.Build.Dense(3);
        bestInliers = new List<int>();
        double bestError = double.MaxValue;

        if (camera1Points.Count != camera2Points.Count || camera1Points.Count < 8)
            return;

        List<VectorD> camera1NormalizedCameraCoordinates = CVProjection.IntrinsicUnProjectPoints(camera1Points, K1, d1);
        List<VectorD> camera2NormalizedCameraCoordinates = CVProjection.IntrinsicUnProjectPoints(camera2Points, K2, d2);

        // Hartley Normalize
        List<VectorD> points1 = CVHomography.HartleyNormalize(camera1NormalizedCameraCoordinates, out MatrixD T1);
        List<VectorD> points2 = CVHomography.HartleyNormalize(camera2NormalizedCameraCoordinates, out MatrixD T2);

        Random random = new Random();
        int n = camera1NormalizedCameraCoordinates.Count;
        for (int iter = 0; iter < iterations; iter++)
        {
            // 1. Pick 8 random matches
            HashSet<int> indices = new HashSet<int>();

            while (indices.Count < 8)
                indices.Add(random.Next(n));

            List<VectorD> srcPointsSample = new();
            List<VectorD> dstPointsSample = new();

            foreach (int i in indices)
            {
                srcPointsSample.Add(points1[i]);
                dstPointsSample.Add(points2[i]);
            }

            MatrixD En = EstimateEssential(srcPointsSample, dstPointsSample);

            // Undo Hartley normalization
            MatrixD E = CVHomography.HartleyDenormalize(En, T1, T2);
            E = EnforceEssential(E);

            double error = MeanSampsonErrorInliers(camera1NormalizedCameraCoordinates, camera2NormalizedCameraCoordinates, E, threshold, out List<int> inliers);

            // 3. Keep best model
            if (inliers.Count > bestInliers.Count || (inliers.Count == bestInliers.Count && error < bestError))
            {
                bestInliers = inliers;
                bestError = error;
            }
        }

        // Console.WriteLine($"Sampson Error {bestError}");
        // Console.WriteLine($"Inliers {bestInliers.Count} / {n}");

        if (bestInliers.Count < 8)
            throw new Exception("No valid Essential Matrix found");

        // 4. Recalculate using all inliers
        List<VectorD> inlierSrc = new();
        List<VectorD> inlierDst = new();

        foreach (int i in bestInliers)
        {
            inlierSrc.Add(camera1Points[i]);
            inlierDst.Add(camera2Points[i]);
        }

        double inlierError = EstimateCameraPose(inlierSrc, inlierDst, K1, K2, d1, d2, out R, out t);

        // Console.WriteLine($"Final Sampson Error: {inlierError}");
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
            EstimateEssential(
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

        int pose =
            SelectPose(
                rotations,
                translations,
                inlierP1,
                inlierP2);

        R = rotations[pose];
        t = translations[pose];

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
            VectorD X = triangulate(R, t, normalizedPixels1[i], normalizedPixels2[i]);
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
        EstimateCameraPoseRansac(src, dst, K1, K2, d1, d2, iterations, threshold, out R, out t, out List<int> inliers);
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