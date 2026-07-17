using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace CVNet;

public class CVCamera
{
    public static List<Vector<double>>
    Normalize(List<Vector<double>> pts, out Matrix<double> T)
    {
        double cx = pts.Average(p => p[0]);
        double cy = pts.Average(p => p[1]);

        double meanDist = pts
            .Average(p =>
            {
                double dx = p[0] - cx;
                double dy = p[1] - cy;
                return Math.Sqrt(dx * dx + dy * dy);
            });

        double s = Math.Sqrt(2) / meanDist;

        T = DenseMatrix.OfArray(new double[,]
        {
            { s, 0, -s * cx },
            { 0, s, -s * cy },
            { 0, 0, 1 }
        });

        var norm = new List<Vector<double>>();
        foreach (var p in pts)
            norm.Add(DenseVector.OfArray([s * (p[0] - cx), s * (p[1] - cy)]));

        return norm;
    }

    public static Matrix<double> Denormalize(
        Matrix<double> Hn,
        Matrix<double> Timg,
        Matrix<double> Twrld)
    {
        var TimgInv = Timg.Inverse();

        return TimgInv * Hn * Twrld;
    }

    public static Matrix<double> ComputeHomography(List<Vector<double>> img, List<Vector<double>> wrld)
    {
        DenseMatrix A = DenseMatrix.Create(2 * wrld.Count, 9, 0);

        for (int i = 0; i < wrld.Count; i++)
        {
            double X = wrld[i][0];
            double Y = wrld[i][1];
            double x = img[i][0];
            double y = img[i][1];

            A[2 * i, 0] = -X;
            A[2 * i, 1] = -Y;
            A[2 * i, 2] = -1;
            A[2 * i, 6] = x * X;
            A[2 * i, 7] = x * Y;
            A[2 * i, 8] = x;

            A[2 * i + 1, 3] = -X;
            A[2 * i + 1, 4] = -Y;
            A[2 * i + 1, 5] = -1;
            A[2 * i + 1, 6] = y * X;
            A[2 * i + 1, 7] = y * Y;
            A[2 * i + 1, 8] = y;
        }

        var svd = A.Svd(true);
        Vector<double> h = svd.VT.Row(svd.VT.RowCount - 1);

        DenseMatrix H = DenseMatrix.OfArray(new double[,]
        {
        { h[0], h[1], h[2] },
        { h[3], h[4], h[5] },
        { h[6], h[7], h[8] }
        });

        return H / H[2, 2];
    }

    public static Matrix<double> ComputeHomographyStable(List<Vector<double>> src, List<Vector<double>> dst)
    {
        if (src.Count < 4)
            return DenseMatrix.OfArray(new double[,]
            {
                { 1, 0, 0 },
                { 0, 1, 0 },
                {0,0, 1 }
            });

        List<Vector<double>> normalizedSrcPoints = Normalize(src, out var TSrc);
        List<Vector<double>> normalizedDstPoints = Normalize(dst, out var TDst);

        Matrix<double> homography = ComputeHomography(normalizedSrcPoints, normalizedDstPoints);

        return Denormalize(homography, TSrc, TDst);
    }

    public static Matrix<double> ComputeHomographyRansac(
        List<Vector<double>> src,
        List<Vector<double>> dst,
        int iterations,
        double threshold,
        out List<int> bestInliers)
    {
        bestInliers = new List<int>();
        Matrix<double> bestH = Matrix<double>.Build.DenseIdentity(3);

        if (src.Count != dst.Count || src.Count < 4)
            return bestH;

        Random random = new Random();

        int n = src.Count;

        for (int iter = 0; iter < iterations; iter++)
        {
            // 1. Pick 4 random matches
            HashSet<int> indices = new HashSet<int>();

            while (indices.Count < 4)
                indices.Add(random.Next(n));

            List<Vector<double>> srcSample = new();
            List<Vector<double>> dstSample = new();

            foreach (int i in indices)
            {
                srcSample.Add(src[i]);
                dstSample.Add(dst[i]);
            }

            Matrix<double> H;

            try
            {
                H = CVWarp.GetPerspectiveTransform(
                    srcSample,
                    dstSample);
            }
            catch
            {
                continue;
            }

            // 2. Count inliers
            List<int> inliers = new();

            for (int i = 0; i < n; i++)
            {
                Vector<double> projected = CVWarp.TransformPoint(src[i], H);

                double dx = projected[0] - dst[i][0];
                double dy = projected[1] - dst[i][1];
                double error = Math.Sqrt(dx * dx + dy * dy);

                if (error < threshold)
                    inliers.Add(i);
            }


            // 3. Keep best model
            if (inliers.Count > bestInliers.Count)
            {
                bestInliers = inliers;
                bestH = H;
            }
        }

        if (bestInliers.Count < 4)
            throw new Exception("No valid homography found");


        // 4. Recalculate using all inliers
        List<Vector<double>> finalSrc = new();
        List<Vector<double>> finalDst = new();

        foreach (int i in bestInliers)
        {
            finalSrc.Add(src[i]);
            finalDst.Add(dst[i]);
        }

        return ComputeHomographyStable(finalSrc, finalDst);
    }

    private static Vector<double> Vij(Matrix<double> H, int i, int j)
    {
        double[] h = H.ToRowMajorArray();

        double hi1 = H[0, i];
        double hi2 = H[1, i];
        double hi3 = H[2, i];

        double hj1 = H[0, j];
        double hj2 = H[1, j];
        double hj3 = H[2, j];

        return DenseVector.OfArray(new[]
        {
            hi1 * hj1,
            hi1 * hj2 + hi2 * hj1,
            hi2 * hj2,
            hi3 * hj1 + hi1 * hj3,
            hi3 * hj2 + hi2 * hj3,
            hi3 * hj3
        });
    }

    private static Vector<double> ViiMinusVjj(Matrix<double> H)
    {
        Vector<double> v11 = Vij(H, 0, 0);
        Vector<double> v22 = Vij(H, 1, 1);

        return v11 - v22;
    }

    public static Matrix<double> ComputeIntrinsics(List<Matrix<double>> homographies)
    {
        DenseMatrix V = DenseMatrix.Create(homographies.Count * 2, 6, 0);

        int row = 0;

        foreach (var H in homographies)
        {
            var v12 = Vij(H, 0, 1);
            var v11_v22 = ViiMinusVjj(H);

            V.SetRow(row++, v12);
            V.SetRow(row++, v11_v22);
        }

        if (V.ColumnAbsoluteSums().Any(v => double.IsNaN(v) || double.IsInfinity(v)))
            throw new Exception("V contains NaN/Inf before SVD");

        var svd = V.Svd(true);

        if (svd.S.Min() < 1e-12)
            throw new Exception("V is rank-deficient (bad homographies)");

        Vector<double> b = svd.VT.Row(5);

        if (b.Any(x => double.IsNaN(x) || double.IsInfinity(x)))
            throw new Exception("SVD produced invalid solution");

        double B11 = b[0];
        double B12 = b[1];
        double B22 = b[2];
        double B13 = b[3];
        double B23 = b[4];
        double B33 = b[5];

        double v0 =
            (B12 * B13 - B11 * B23) /
            (B11 * B22 - B12 * B12);

        double lambda =
            B33 -
            (B13 * B13 +
             v0 * (B12 * B13 - B11 * B23))
            / B11;

        double alpha =
            Math.Sqrt(lambda / B11);

        double beta =
            Math.Sqrt(
                lambda * B11 /
                (B11 * B22 - B12 * B12));

        double gamma =
            -B12 * alpha * alpha * beta / lambda;

        double u0 =
            gamma * v0 / beta -
            B13 * alpha * alpha / lambda;

        double denom = (B11 * B22 - B12 * B12);
        if (Math.Abs(denom) < 1e-14)
            throw new Exception("Degenerate intrinsic system");

        double cy = (B12 * B13 - B11 * B23) / denom;
        double lam = B33 - (B13 * B13 + cy * (B12 * B13 - B11 * B23)) / B11;

        double v = lam / B11;

        if (v <= 0 || double.IsNaN(v))
            throw new Exception("Invalid fx sqrt input");

        double fx = Math.Sqrt(v);
        double fy = Math.Sqrt(v / denom);
        double cx = -B13 * fx * fx / lam;
        double cy2 = cy;

        return DenseMatrix.OfArray(new double[,]
        {
            { alpha, gamma, u0 },
            { 0, beta, v0 },
            { 0, 0, 1 }
        });
    }

    static Vector<double> Cross(
        Vector<double> a,
        Vector<double> b)
    {
        return DenseVector.OfArray(new[]
        {
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    });
    }

    public static void ComputeExtrinsics(Matrix<double> H, Matrix<double> Kinv, out Matrix<double> R, out Vector<double> t)
    {
        var B = Kinv * H;

        var b1 = B.Column(0);
        var b2 = B.Column(1);
        var b3 = B.Column(2);

        double lambda =
            2.0 /
            (b1.L2Norm() + b2.L2Norm());

        var r1 = lambda * b1;
        var r2 = lambda * b2;
        t = lambda * b3;

        var r3 = Cross(r1, r2);

        R = DenseMatrix.OfColumnVectors(
            r1,
            r2,
            r3);

        var svd = R.Svd(true);
        var U = svd.U;
        var Vt = svd.VT;

        var D = DenseMatrix.CreateIdentity(3);

        if ((U * Vt).Determinant() < 0)
            D[2, 2] = -1;

        R = U * D * Vt;
    }

    public static void ComputeExtrinsics(List<Matrix<double>> homographies, Matrix<double> K, out List<Matrix<double>> Rs, out List<Vector<double>> ts)
    {
        var Kinv = K.Inverse();

        Rs = new List<Matrix<double>>();
        ts = new List<Vector<double>>();

        foreach (var H in homographies)
        {
            ComputeExtrinsics(H, Kinv, out Matrix<double> R, out Vector<double> t);

            Rs.Add(R);
            ts.Add(t);
        }
    }

    public static List<List<Vector<double>>> TransformPoints(List<List<Vector<double>>> points, List<Matrix<double>> Rs, List<Vector<double>> ts)
    {
        List<List<Vector<double>>> transformedPoints = new List<List<Vector<double>>>();

        if (points.Count != Rs.Count || points.Count != ts.Count) return transformedPoints;

        for (int i = 0; i < points.Count; i++)
        {
            transformedPoints.Add(new List<Vector<double>>());

            foreach (Vector<double> point in points[i])
            {
                transformedPoints[i].Add(Rs[i] * point + ts[i]);
            }
        }

        return transformedPoints;
    }

    private static Vector<double> Project(
    Vector<double> P,
    Matrix<double> K,
    Vector<double> dist)
    {
        double fx = K[0, 0], fy = K[1, 1];
        double cx = K[0, 2], cy = K[1, 2];

        double x = P[0] / P[2];
        double y = P[1] / P[2];

        double r2 = x * x + y * y;

        double k1 = dist[0], k2 = dist[1];
        double p1 = dist[2], p2 = dist[3];

        double radial = 1 + k1 * r2 + k2 * r2 * r2;

        double xDist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
        double yDist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;

        return DenseVector.OfArray(new double[]
        {
        fx*xDist + cx,
        fy*yDist + cy
        });
    }

    public static double ComputeError(
    List<Vector<double>> world,
    List<Vector<double>> image,
    Matrix<double> K,
    Vector<double> dist)
    {
        double error = 0;

        for (int i = 0; i < world.Count; i++)
        {
            Vector<double> proj = Project(world[i], K, dist);

            double dx = proj[0] - image[i][0];
            double dy = proj[1] - image[i][1];

            error += dx * dx + dy * dy;
        }

        return error / world.Count;
    }

    private static void
    ProjectWithJacobian(Vector<double> P, Matrix<double> K, Vector<double> d, out Vector<double> proj, out Matrix<double> J)
    {
        double fx = K[0, 0];
        double fy = K[1, 1];
        double cx = K[0, 2];
        double cy = K[1, 2];

        double x = P[0] / P[2];
        double y = P[1] / P[2];

        double r2 = x * x + y * y;
        double r4 = r2 * r2;

        double k1 = d[0];
        double k2 = d[1];
        double p1 = d[2];
        double p2 = d[3];
        double k3 = d[4];

        double radial = 1 + k1 * r2 + k2 * r4 + k3 * r4 * r2;

        double xDist =
            x * radial +
            2 * p1 * x * y +
            p2 * (r2 + 2 * x * x);

        double yDist =
            y * radial +
            p1 * (r2 + 2 * y * y) +
            2 * p2 * x * y;

        double u = fx * xDist + cx;
        double v = fy * yDist + cy;

        proj = DenseVector.OfArray(new double[] { u, v });

        J = DenseMatrix.Create(2, 5, 0);

        // --- partial derivatives (simplified but correct structure) ---

        double dr_dx = x * (2 * k1 + 4 * k2 * r2 + 6 * k3 * r4);
        double dr_dy = y * (2 * k1 + 4 * k2 * r2 + 6 * k3 * r4);

        // k1
        J[0, 0] = fx * x * r2;
        J[1, 0] = fy * y * r2;

        // k2
        J[0, 1] = fx * x * r4;
        J[1, 1] = fy * y * r4;

        // p1
        J[0, 2] = fx * (2 * x * y);
        J[1, 2] = fy * (r2 + 2 * y * y);

        // p2
        J[0, 3] = fx * (r2 + 2 * x * x);
        J[1, 3] = fy * (2 * x * y);

        // k3
        J[0, 4] = fx * x * r4 * r2;
        J[1, 4] = fy * y * r4 * r2;
    }

    private static void
    ComputeResidualAndJacobian(
        List<Vector<double>> world,
        List<Vector<double>> image,
        Matrix<double> K,
        Vector<double> d,
        out DenseVector r,
        out DenseMatrix J)
    {
        int n = world.Count;

        r = DenseVector.Create(2 * n, 0);
        J = DenseMatrix.Create(2 * n, d.Count, 0);

        for (int i = 0; i < n; i++)
        {
            ProjectWithJacobian(world[i], K, d, out Vector<double> proj, out Matrix<double> jac);

            double dx = proj[0] - image[i][0];
            double dy = proj[1] - image[i][1];

            r[2 * i] = dx;
            r[2 * i + 1] = dy;

            for (int k = 0; k < d.Count; k++)
            {
                J[2 * i, k] = jac[0, k];
                J[2 * i + 1, k] = jac[1, k];
            }
        }
    }

    public static Vector<double> LevenbergMarquardt(
        List<Vector<double>> world,
        List<Vector<double>> image,
        Matrix<double> K,
        Vector<double> dist,
        int maxIter = 30)
    {
        double lambda = 1e-3;

        Vector<double> x = dist.Clone();

        for (int iter = 0; iter < maxIter; iter++)
        {
            ComputeResidualAndJacobian(world, image, K, x, out DenseVector r, out DenseMatrix J);

            var JT = J.Transpose();
            var A = JT * J;

            var g = JT * r;

            // damping (LM)
            for (int i = 0; i < A.RowCount; i++)
                A[i, i] *= (1.0 + lambda);

            var h = A.Svd(true).Solve(-g);

            Vector<double> xNew = DenseVector.OfArray(new double[] { 0, 0, 0, 0, 0 });
            for (int i = 0; i < x.Count; i++)
                xNew[i] = x[i] + h[i];

            ComputeResidualAndJacobian(world, image, K, xNew, out DenseVector rNew, out DenseMatrix _);

            double err = r.DotProduct(r);
            double errNew = rNew.DotProduct(rNew);

            if (errNew < err)
            {
                x = xNew;
                lambda *= 0.7;
            }
            else
            {
                lambda *= 2.0;
            }
        }

        return x;
    }

    private static List<Vector<double>> NormalizeCameraPoints(List<Vector<double>> pixels, Matrix<double> K)
    {
        var Kinv = K.Inverse();

        var result = new List<Vector<double>>();

        foreach (var p in pixels)
        {
            var hp = DenseVector.OfArray([p[0], p[1], 1.0]);

            var x = Kinv * hp;

            result.Add(DenseVector.OfArray([x[0] / x[2], x[1] / x[2], 1.0]));
        }

        return result;
    }


    private static Matrix<double> EnforceEssential(
        Matrix<double> E)
    {
        var svd = E.Svd(true);

        var U = svd.U;
        var Vt = svd.VT;

        // Ensure proper rotations
        if (U.Determinant() < 0)
        {
            for (int i = 0; i < 3; i++)
                U[i, 2] *= -1;
        }

        if (Vt.Determinant() < 0)
        {
            for (int i = 0; i < 3; i++)
                Vt[2, i] *= -1;
        }

        double s = (svd.S[0] + svd.S[1]) * 0.5;

        var S = DenseMatrix.OfArray(new double[,] {
            {s,0,0},
            {0,s,0},
            {0,0,0}
        });

        return U * S * Vt;
    }


    private static Matrix<double> EstimateEssential(
        List<Vector<double>> p1,
        List<Vector<double>> p2)
    {
        int n = p1.Count;

        if (n < 8) throw new Exception("Essential matrix requires at least 8 points");

        var A = DenseMatrix.Create(n, 9, 0);

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
        Vector<double> e = svd.VT.Row(8);

        var E = DenseMatrix.OfArray(new double[,] {
            {e[0],e[1],e[2]},
            {e[3],e[4],e[5]},
            {e[6],e[7],e[8]}
        });

        return EnforceEssential(E);
    }

    private static void DecomposeEssential(
       Matrix<double> E,
       out List<Matrix<double>> rotations,
       out List<Vector<double>> translations)
    {
        var svd = E.Svd(true);

        var U = svd.U;
        var Vt = svd.VT;

        // Fix SVD signs
        if (U.Determinant() < 0)
            U.SetColumn(2, -U.Column(2));

        if (Vt.Determinant() < 0)
            Vt.SetRow(2, -Vt.Row(2));

        var W = DenseMatrix.OfArray(new double[,] {
            {0,-1,0},
            {1, 0,0},
            {0, 0,1}
        });


        Matrix<double> R1 = U * W * Vt;
        Matrix<double> R2 = U * W.Transpose() * Vt;

        Vector<double> t = U.Column(2);

        rotations = new List<Matrix<double>> { R1, R1, R2, R2 };
        translations = new List<Vector<double>> { t, -t, t, -t };
    }



    private static Vector<double> Triangulate(
        Matrix<double> R,
        Vector<double> t,
        Vector<double> x1,
        Vector<double> x2)
    {
        //Camera 1: P1 = [I | 0]
        //Camera 2: P2 = [R | t]

        var P1 = DenseMatrix.OfArray(new double[,]
        {
            {1,0,0,0},
            {0,1,0,0},
            {0,0,1,0}
        });

        var P2 = DenseMatrix.Create(3, 4, 0);
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
                P2[r, c] = R[r, c];

            P2[r, 3] = t[r];
        }

        var A = DenseMatrix.Create(4, 4, 0);

        // first camera
        A.SetRow(0, x1[0] * P1.Row(2) - P1.Row(0));
        A.SetRow(1, x1[1] * P1.Row(2) - P1.Row(1));

        // second camera
        A.SetRow(2, x2[0] * P2.Row(2) - P2.Row(0));
        A.SetRow(3, x2[1] * P2.Row(2) - P2.Row(1));

        var svd = A.Svd(true);

        Vector<double> X = svd.VT.Row(3);

        if (Math.Abs(X[3]) < 1e-12)
            return DenseVector.OfArray([double.NaN, double.NaN, double.NaN]);

        return DenseVector.OfArray([X[0] / X[3], X[1] / X[3], X[2] / X[3]]);
    }

    private static int SelectPose(
        List<Matrix<double>> rotations,
        List<Vector<double>> translations,
        List<Vector<double>> p1,
        List<Vector<double>> p2)
    {
        int bestIndex = -1;
        int bestCount = -1;

        for (int pose = 0; pose < 4; pose++)
        {
            int count = 0;

            for (int i = 0; i < p1.Count; i++)
            {
                Vector<double> X = Triangulate(rotations[pose], translations[pose], p1[i], p2[i]);

                if (double.IsNaN(X[0]))
                    continue;

                double z1 = X[2];

                Vector<double> X2 = rotations[pose] * X + translations[pose];

                double z2 = X2[2];

                if (z1 > 0 && z2 > 0)
                    count++;
            }

            if (count > bestCount)
            {
                bestCount = count;
                bestIndex = pose;
            }
        }

        if (bestIndex < 0)
            throw new Exception("No valid camera pose found");

        return bestIndex;
    }

    private static (List<Vector<double>>, Matrix<double> T) HartleyNormalize(
        List<Vector<double>> points)
    {
        int n = points.Count;

        double cx = 0;
        double cy = 0;

        // Compute centroid
        foreach (var p in points)
        {
            cx += p[0];
            cy += p[1];
        }

        cx /= n;
        cy /= n;

        // Average distance from centroid
        double meanDistance = 0;

        foreach (var p in points)
        {
            double dx = p[0] - cx;
            double dy = p[1] - cy;

            meanDistance += Math.Sqrt(dx * dx + dy * dy);
        }

        meanDistance /= n;

        // Scale so average distance = sqrt(2)
        double s = Math.Sqrt(2.0) / meanDistance;

        var T = DenseMatrix.OfArray(new double[,] {
            {s, 0, -s * cx},
            {0, s, -s * cy},
            {0, 0, 1}
        });


        var normalized = new List<Vector<double>>();

        foreach (var p in points)
        {
            Vector<double> hp = DenseVector.OfArray([p[0], p[1], 1.0]);
            Vector<double> pn = T * hp;
            normalized.Add(DenseVector.OfArray([pn[0] / pn[2], pn[1] / pn[2], 1.0]));
        }

        return (normalized, T);
    }

    public static void EstimateCamera2Pose(
       List<Vector<double>> normalizedPixels1,
       List<Vector<double>> normalizedPixels2,
       out Matrix<double> R,
       out Vector<double> t)
    {
        // Estimate essential matrix
        var h1 = HartleyNormalize(normalizedPixels1);
        var h2 = HartleyNormalize(normalizedPixels2);

        Matrix<double> En = EstimateEssential(h1.Item1, h2.Item1);

        // Undo Hartley normalization
        Matrix<double> E = h2.T.Transpose() * En * h1.T;

        E = EnforceEssential(E);

        // Four possible poses
        DecomposeEssential(E, out var rotations, out var translations);

        // Pick the pose where points are in front
        // of both cameras
        int index = SelectPose(rotations, translations, normalizedPixels1, normalizedPixels2);

        R = rotations[index];
        t = translations[index];
    }

    public static List<Vector<double>> TriangulateAll(
        List<Vector<double>> pixels1,
        List<Vector<double>> pixels2,
        Matrix<double> K1,
        Matrix<double> K2,
        double baseline,
        out Matrix<double> R,
        out Vector<double> t)
    {
        if (pixels1.Count != pixels2.Count)
            throw new ArgumentException(
                "Point lists must have same length");

        List<Vector<double>> normalizedPixels1 = NormalizeCameraPoints(pixels1, K1);
        List<Vector<double>> normalizedPixels2 = NormalizeCameraPoints(pixels2, K2);

        EstimateCamera2Pose(normalizedPixels1, normalizedPixels2, out R, out t);

        var points = new List<Vector<double>>();

        for (int i = 0; i < normalizedPixels1.Count; i++)
        {
            Vector<double> X = Triangulate(R, t, normalizedPixels1[i], normalizedPixels2[i]);
            points.Add(X);
        }

        // Scale translation and reconstructed points
        double scale = baseline / t.L2Norm();
        t *= scale;

        for (int i = 0; i < points.Count; i++)
        {
            points[i] *= scale;
        }

        return points;
    }
};