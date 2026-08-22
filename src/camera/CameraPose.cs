namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using DenseMatrixD = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix;

public static partial class CVCamera
{
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


    private static MatrixD NormalizedEstimateEssential(
        List<VectorD> normalizedImagePoints1,
        List<VectorD> normalizedImagePoints2)
    {
        int n = normalizedImagePoints1.Count;

        if (n < 8) throw new Exception("Essential matrix requires at least 8 points");

        var A = DenseMatrixD.Create(n, 9, 0);

        for (int i = 0; i < n; i++)
        {
            double x1 = normalizedImagePoints1[i][0];
            double y1 = normalizedImagePoints1[i][1];

            double x2 = normalizedImagePoints2[i][0];
            double y2 = normalizedImagePoints2[i][1];

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

    public static VectorD NormalizedTriangulate(
        MatrixD R,
        VectorD t,
        VectorD normalizedImagePoint1,
        VectorD normalizedImagePoint2)
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
        A.SetRow(0, normalizedImagePoint1[0] * P1.Row(2) - P1.Row(0));
        A.SetRow(1, normalizedImagePoint1[1] * P1.Row(2) - P1.Row(1));

        // second camera
        A.SetRow(2, normalizedImagePoint2[0] * P2.Row(2) - P2.Row(0));
        A.SetRow(3, normalizedImagePoint2[1] * P2.Row(2) - P2.Row(1));

        var svd = A.Svd(true);

        VectorD X = svd.VT.Row(3);

        if (Math.Abs(X[3]) < 1e-12)
            return DenseVectorD.OfArray([double.NaN, double.NaN, double.NaN]);

        return DenseVectorD.OfArray([X[0] / X[3], X[1] / X[3], X[2] / X[3]]);
    }

    public static double NormalizedReprojectionError(
        VectorD pixelPoint1,
        VectorD pixelPoint2,
        VectorD normalizedImagePoint1,
        VectorD normalizedImagePoint2,
        MatrixD R,
        VectorD t,
        MatrixD K1,
        MatrixD K2,
        VectorD d1,
        VectorD d2)
    {
        VectorD worldPoint1 = NormalizedTriangulate(R, t, normalizedImagePoint1, normalizedImagePoint2);
        VectorD worldPoint2 = R * worldPoint1 + t;

        // Point behind one camera
        if (worldPoint1[2] <= 10e-5) return double.NaN;
        if (worldPoint2[2] <= 10e-5) return double.NaN;

        VectorD p1 = CVProjection.ProjectPoint(worldPoint1, DenseMatrixD.CreateIdentity(3), DenseVectorD.OfArray([0, 0, 0]), K1, d1);
        VectorD p2 = CVProjection.ProjectPoint(worldPoint1, R, t, K2, d2);

        double dx1 = p1[0] - pixelPoint1[0];
        double dy1 = p1[1] - pixelPoint1[1];
        double dx2 = p2[0] - pixelPoint2[0];
        double dy2 = p2[1] - pixelPoint2[1];
        double errorCam1 = Math.Sqrt(dx1 * dx1 + dy1 * dy1);
        double errorCam2 = Math.Sqrt(dx2 * dx2 + dy2 * dy2);
        return (errorCam1 + errorCam2) * 0.5;
    }

    public static double NormalizedMeanReprojectionError(
        List<VectorD> pixelPoints1,
        List<VectorD> pixelPoints2,
        List<VectorD> normalizedImagePoints1,
        List<VectorD> normalizedImagePoints2,
        MatrixD R,
        VectorD t,
        MatrixD K1,
        MatrixD K2,
        VectorD d1,
        VectorD d2)
    {
        double error = 0.0;
        int validError = 0;

        for (int i = 0; i < normalizedImagePoints1.Count; i++)
        {
            double errorI = NormalizedReprojectionError(pixelPoints1[i], pixelPoints2[i], normalizedImagePoints1[i], normalizedImagePoints2[i], R, t, K1, K2, d1, d2);

            if (double.IsFinite(errorI))
            {
                error += errorI;
                validError++;
            }
        }

        if (validError == 0) return double.NaN;
        return error / validError;
    }

    public static double NormalizedReprojectionErrorThreshold(
            List<VectorD> pixelPoints1,
            List<VectorD> pixelPoints2,
            List<VectorD> normalizedImagePoints1,
            List<VectorD> normalizedImagePoints2,
            MatrixD R,
            VectorD t,
            MatrixD K1,
            MatrixD K2,
            VectorD d1,
            VectorD d2,
            double threshold,
            out List<int> inliers)
    {
        inliers = new List<int>();
        double error = 0.0;

        for (int i = 0; i < normalizedImagePoints1.Count; i++)
        {
            double errorI = NormalizedReprojectionError(pixelPoints1[i], pixelPoints2[i], normalizedImagePoints1[i], normalizedImagePoints2[i], R, t, K1, K2, d1, d2);

            if (double.IsFinite(errorI) && errorI < threshold)
            {
                error += errorI;
                inliers.Add(i);
            }
        }

        if (inliers.Count == 0) return double.NaN;
        return error / inliers.Count;
    }

    public static double MeanReprojectionError(
        List<VectorD> pixelPoints1,
        List<VectorD> pixelPoints2,
        MatrixD R,
        VectorD t,
        MatrixD K1,
        MatrixD K2,
        VectorD d1,
        VectorD d2)
    {
        List<VectorD> normalizedImagePoints1 = CVProjection.IntrinsicUnProjectPoints(pixelPoints1, K1, d1);
        List<VectorD> normalizedImagePoints2 = CVProjection.IntrinsicUnProjectPoints(pixelPoints2, K2, d2);

        return NormalizedMeanReprojectionError(pixelPoints1, pixelPoints2, normalizedImagePoints1, normalizedImagePoints2, R, t, K1, K2, d1, d2);
    }

    private static void NormalizedSelectPose(
        List<MatrixD> Rs,
        List<VectorD> ts,
        List<VectorD> pixelPoints1,
        List<VectorD> pixelPoints2,
        List<VectorD> normalizedImagePoints1,
        List<VectorD> normalizedImagePoints2,
        MatrixD K1,
        MatrixD K2,
        VectorD d1,
        VectorD d2,
        out MatrixD R,
        out VectorD t)
    {
        int bestIndex = -1;
        int bestCount = -1;
        double bestError = double.MaxValue;

        for (int pose = 0; pose < Rs.Count; pose++)
        {
            double error = NormalizedReprojectionErrorThreshold(pixelPoints1, pixelPoints2, normalizedImagePoints1, normalizedImagePoints2, Rs[pose], ts[pose], K1, K2, d1, d2, double.MaxValue, out List<int> inliers);

            if (inliers.Count > bestCount || (inliers.Count == bestCount && error < bestError))
            {
                bestCount = inliers.Count;
                bestIndex = pose;
                bestError = error;
            }
        }

        if (bestIndex < 0)
            throw new Exception("No valid camera pose found");

        R = Rs[bestIndex];
        t = ts[bestIndex];
    }

    public static void HartleyNormalizedEstimateCameraPoses5Point(
        List<VectorD> hartleyNormalizedImagePoints1,
        List<VectorD> hartleyNormalizedImagePoints2,
        MatrixD T1,
        MatrixD T2,
        out List<MatrixD> Es,
        out List<List<MatrixD>> Rss,
        out List<List<VectorD>> tss)
    {
        List<MatrixD> ECandidates = CVHomographySolver.EstimateEssential5Point(hartleyNormalizedImagePoints1, hartleyNormalizedImagePoints2);

        Es = new List<MatrixD>();
        Rss = new List<List<MatrixD>>();
        tss = new List<List<VectorD>>();
        foreach (MatrixD ECandidate in ECandidates)
        {
            MatrixD E = CVHomography.HartleyDenormalize(ECandidate, T1, T2);
            E = EnforceEssential(E);
            Es.Add(E);
            DecomposeEssential(E, out List<MatrixD> Rs, out List<VectorD> ts);
            Rss.Add(Rs);
            tss.Add(ts);
        }
    }


    public static void HartleyNormalizedEstimateCameraPoses(
        List<VectorD> hartleyNormalizedImagePoints1,
        List<VectorD> hartleyNormalizedImagePoints2,
        MatrixD T1,
        MatrixD T2,
        out MatrixD E,
        out List<MatrixD> Rs,
        out List<VectorD> ts)
    {
        E = NormalizedEstimateEssential(hartleyNormalizedImagePoints1, hartleyNormalizedImagePoints2);
        E = CVHomography.HartleyDenormalize(E, T1, T2);
        E = EnforceEssential(E);

        // Four possible poses
        DecomposeEssential(E, out Rs, out ts);
    }

    public static void EstimateCameraPoses(
        List<VectorD> pixelPoints1,
        List<VectorD> pixelPoints2,
        MatrixD K1,
        MatrixD K2,
        VectorD d1,
        VectorD d2,
        out MatrixD T1,
        out MatrixD T2,
        out MatrixD E,
        out List<MatrixD> Rs,
        out List<VectorD> ts)
    {
        List<VectorD> normalizedImagePoints1 = CVProjection.IntrinsicUnProjectPoints(pixelPoints1, K1, d1);
        List<VectorD> normalizedImagePoints2 = CVProjection.IntrinsicUnProjectPoints(pixelPoints2, K2, d2);

        List<VectorD> hartleyNormalizedImagePoints1 = CVHomography.HartleyNormalize(normalizedImagePoints1, out T1);
        List<VectorD> hartleyNormalizedImagePoints2 = CVHomography.HartleyNormalize(normalizedImagePoints2, out T2);

        HartleyNormalizedEstimateCameraPoses(hartleyNormalizedImagePoints1, hartleyNormalizedImagePoints2, T1, T2, out E, out Rs, out ts);
    }

    public static double NormalizedSampsonError(
                VectorD normalizedImagePoint1,
                VectorD normalizedImagePoint2,
                MatrixD E,
                MatrixD Et)
    {
        VectorD Ex1 = E * normalizedImagePoint1;
        VectorD Etx2 = Et * normalizedImagePoint2;

        double numer = normalizedImagePoint2.DotProduct(Ex1);
        double denom = Ex1[0] * Ex1[0] + Ex1[1] * Ex1[1] + Etx2[0] * Etx2[0] + Etx2[1] * Etx2[1];

        if (denom < 1e-12)
            return double.NaN;

        return numer * numer / denom;
    }

    public static double NormalizedMeanSampsonError(
            List<VectorD> srcNorm,
            List<VectorD> dstNorm,
            MatrixD E)
    {
        double error = 0.0;
        int validError = 0;

        MatrixD Et = E.Transpose();

        for (int i = 0; i < srcNorm.Count; i++)
        {
            double errorI = NormalizedSampsonError(srcNorm[i], dstNorm[i], E, Et);

            if (double.IsFinite(errorI))
            {
                error += errorI;
                validError++;
            }
        }

        if (validError == 0) return double.NaN;
        return error / validError;
    }

    public static double NormalizedSampsonErrorThreshold(
        List<VectorD> normalizedImagePoints1,
        List<VectorD> normalizedImagePoints2,
        MatrixD E,
        double threshold,
        out List<int> inliers)
    {
        inliers = new();
        double error = 0.0;

        MatrixD Et = E.Transpose();

        for (int i = 0; i < normalizedImagePoints1.Count; i++)
        {
            double errorI = NormalizedSampsonError(normalizedImagePoints1[i], normalizedImagePoints2[i], E, Et);

            if (double.IsFinite(errorI) && errorI < threshold)
            {
                error += errorI;
                inliers.Add(i);
            }

        }

        if (inliers.Count == 0) return double.NaN;
        return error / inliers.Count;
    }

    public static double EstimateCameraPoseRansac(
            List<VectorD> pixelPoints1,
            List<VectorD> pixelPoints2,
            MatrixD K1,
            MatrixD K2,
            VectorD d1,
            VectorD d2,
            int iterations,
            double threshold,
            out MatrixD E,
            out MatrixD R,
            out VectorD t,
            out List<int> bestInliers)
    {

        double f1 = Math.Sqrt(K1[0, 0] * K1[1, 1]);
        double f2 = Math.Sqrt(K2[0, 0] * K2[1, 1]);
        double sampsonThreshold = threshold * threshold / (f1 * f2);

        Console.WriteLine(sampsonThreshold);

        R = DenseMatrixD.CreateIdentity(3);
        t = DenseVectorD.Build.Dense(3);
        bestInliers = new List<int>();
        double bestError = double.MaxValue;

        if (pixelPoints1.Count != pixelPoints2.Count || pixelPoints1.Count < 8)
            throw new Exception("Invalid Points");

        // Normalize Camera Coordinates
        List<VectorD> normalizedImagePoints1 = CVProjection.IntrinsicUnProjectPoints(pixelPoints1, K1, d1);
        List<VectorD> normalizedImagePoints2 = CVProjection.IntrinsicUnProjectPoints(pixelPoints2, K2, d2);

        List<VectorD> hartleyNormalizedImagePoints1 = CVHomography.HartleyNormalize(normalizedImagePoints1, out MatrixD T1);
        List<VectorD> hartleyNormalizedImagePoints2 = CVHomography.HartleyNormalize(normalizedImagePoints2, out MatrixD T2);

        Random random = new Random();
        int n = normalizedImagePoints1.Count;
        for (int iter = 0; iter < iterations; iter++)
        {
            // Pick 8 random matches
            HashSet<int> indices = new HashSet<int>();

            while (indices.Count < 8)
                indices.Add(random.Next(n));

            List<VectorD> pixelPointsSample1 = new();
            List<VectorD> pixelPointsSample2 = new();

            List<VectorD> hartleyNormalizedImagePointsSample1 = new();
            List<VectorD> hartleyNormalizedImagePointsSample2 = new();

            foreach (int i in indices)
            {
                pixelPointsSample1.Add(pixelPoints1[i]);
                pixelPointsSample2.Add(pixelPoints2[i]);

                hartleyNormalizedImagePointsSample1.Add(hartleyNormalizedImagePoints1[i]);
                hartleyNormalizedImagePointsSample2.Add(hartleyNormalizedImagePoints2[i]);
            }

            // Estimate Camera Pose from these Points
            HartleyNormalizedEstimateCameraPoses(hartleyNormalizedImagePointsSample1, hartleyNormalizedImagePointsSample2, T1, T2, out MatrixD EI, out List<MatrixD> RIs, out List<VectorD> tIs);

            // Rate Estimated Camera Pose
            double error = NormalizedSampsonErrorThreshold(normalizedImagePoints1, normalizedImagePoints2, EI, sampsonThreshold, out List<int> inliers);

            if (!double.IsFinite(error)) continue;

            // Keep track of best estimation
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

        // Group all Inliers
        List<VectorD> pixelPointsInliers1 = new();
        List<VectorD> pixelPointsInliers2 = new();

        List<VectorD> normalizedImagePointsInliers1 = new();
        List<VectorD> normalizedImagePointsInliers2 = new();

        foreach (int i in bestInliers)
        {
            pixelPointsInliers1.Add(pixelPoints1[i]);
            pixelPointsInliers2.Add(pixelPoints2[i]);

            normalizedImagePointsInliers1.Add(normalizedImagePoints1[i]);
            normalizedImagePointsInliers2.Add(normalizedImagePoints2[i]);
        }

        List<VectorD> hartleyNormalizedImagePointsInliers1 = CVHomography.HartleyNormalize(normalizedImagePointsInliers1, out MatrixD T1Inliers);
        List<VectorD> hartleyNormalizedImagePointsInliers2 = CVHomography.HartleyNormalize(normalizedImagePointsInliers2, out MatrixD T2Inliers);

        // Calculate Camera Pose from all Inliers
        HartleyNormalizedEstimateCameraPoses(hartleyNormalizedImagePointsInliers1, hartleyNormalizedImagePointsInliers2, T1Inliers, T2Inliers, out E, out List<MatrixD> Rs, out List<VectorD> ts);

        NormalizedSelectPose(Rs, ts, pixelPoints1, pixelPoints2, normalizedImagePoints1, normalizedImagePoints2, K1, K2, d1, d2, out R, out t);

        // Rate Estimated Camera Pose
        return NormalizedMeanSampsonError(normalizedImagePointsInliers1, normalizedImagePointsInliers2, E);
    }
};