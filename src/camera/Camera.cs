namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using DenseMatrixD = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix;

public static class CVCamera
{
    public static List<VectorD> Normalize(List<VectorD> points, out MatrixD T)
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

    public static MatrixD Denormalize(
        MatrixD H,
        MatrixD TImagePoints,
        MatrixD TWorldPoints)
    {
        return TImagePoints.Inverse() * H * TWorldPoints;
    }

    private static MatrixD computeHomography(List<VectorD> imagePoints, List<VectorD> worldPoints)
    {
        DenseMatrixD A = DenseMatrixD.Create(2 * worldPoints.Count, 9, 0);

        for (int i = 0; i < worldPoints.Count; i++)
        {
            double X = worldPoints[i][0];
            double Y = worldPoints[i][1];
            double x = imagePoints[i][0];
            double y = imagePoints[i][1];

            A[2 * i + 0, 0] = -X;
            A[2 * i + 0, 1] = -Y;
            A[2 * i + 0, 2] = -1;
            A[2 * i + 0, 6] = x * X;
            A[2 * i + 0, 7] = x * Y;
            A[2 * i + 0, 8] = x;

            A[2 * i + 1, 3] = -X;
            A[2 * i + 1, 4] = -Y;
            A[2 * i + 1, 5] = -1;
            A[2 * i + 1, 6] = y * X;
            A[2 * i + 1, 7] = y * Y;
            A[2 * i + 1, 8] = y;
        }

        var svd = A.Svd(true);
        VectorD h = svd.VT.Row(svd.VT.RowCount - 1);

        return DenseMatrixD.OfArray(new double[,]
        {
            { h[0], h[1], h[2] },
            { h[3], h[4], h[5] },
            { h[6], h[7], h[8] }
        });
    }

    public static MatrixD ComputeHomography(List<VectorD> imagePoints, List<VectorD> worldPoints)
    {
        if (imagePoints.Count < 4)
            return DenseMatrixD.OfArray(new double[,]
            {
                { 1, 0, 0 },
                { 0, 1, 0 },
                { 0, 0, 1 }
            });

        List<VectorD> normalizedImagePoints = Normalize(imagePoints, out var TImagePoints);
        List<VectorD> normalizedWorldPoints = Normalize(worldPoints, out var TWorldPoints);

        MatrixD normalizedHomography = computeHomography(normalizedImagePoints, normalizedWorldPoints);

        MatrixD homography = Denormalize(normalizedHomography, TImagePoints, TWorldPoints);

        return homography / homography[2, 2];
    }

    public static MatrixD ComputeHomographyRansac(
        List<VectorD> src,
        List<VectorD> dst,
        int iterations,
        double threshold,
        out List<int> bestInliers)
    {
        bestInliers = new List<int>();
        double bestError = double.MaxValue;

        if (src.Count != dst.Count || src.Count < 4)
            return MatrixD.Build.DenseIdentity(3);

        Random random = new Random();
        int n = src.Count;
        for (int iter = 0; iter < iterations; iter++)
        {
            // 1. Pick 4 random matches
            HashSet<int> indices = new HashSet<int>();

            while (indices.Count < 4)
                indices.Add(random.Next(n));

            List<VectorD> srcSample = new();
            List<VectorD> dstSample = new();

            foreach (int i in indices)
            {
                srcSample.Add(src[i]);
                dstSample.Add(dst[i]);
            }

            MatrixD H = CVWarp.GetPerspectiveTransform(srcSample, dstSample);

            // 2. Count inliers
            List<int> inliers = new();
            double error = 0.0;
            for (int i = 0; i < n; i++)
            {
                VectorD projected = CVWarp.TransformPoint(src[i], H);

                double dx = projected[0] - dst[i][0];
                double dy = projected[1] - dst[i][1];
                double homographyError = Math.Sqrt(dx * dx + dy * dy);

                if (homographyError < threshold)
                {
                    error += homographyError;
                    inliers.Add(i);
                }
            }

            // 3. Keep best model
            if (inliers.Count > bestInliers.Count || (inliers.Count == bestInliers.Count && error < bestError))
            {
                bestInliers = inliers;
                bestError = error;
            }
        }

        if (bestInliers.Count < 4)
            throw new Exception("No valid homography found");

        // 4. Recalculate using all inliers
        List<VectorD> finalSrc = new();
        List<VectorD> finalDst = new();

        foreach (int i in bestInliers)
        {
            finalSrc.Add(src[i]);
            finalDst.Add(dst[i]);
        }

        return ComputeHomography(finalSrc, finalDst);
    }

    private static VectorD Vij(MatrixD H, int i, int j)
    {
        double[] h = H.ToRowMajorArray();

        double hi1 = H[0, i];
        double hi2 = H[1, i];
        double hi3 = H[2, i];

        double hj1 = H[0, j];
        double hj2 = H[1, j];
        double hj3 = H[2, j];

        return DenseVectorD.OfArray(new[]
        {
            hi1 * hj1,
            hi1 * hj2 + hi2 * hj1,
            hi2 * hj2,
            hi3 * hj1 + hi1 * hj3,
            hi3 * hj2 + hi2 * hj3,
            hi3 * hj3
        });
    }

    private static VectorD ViiMinusVjj(MatrixD H)
    {
        VectorD v11 = Vij(H, 0, 0);
        VectorD v22 = Vij(H, 1, 1);

        return v11 - v22;
    }

    public static MatrixD ComputeIntrinsics(List<MatrixD> homographies)
    {
        DenseMatrixD V = DenseMatrixD.Create(homographies.Count * 2, 6, 0);

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
        var S = svd.S;

        if (S[4] < 1e-14)
        {
            foreach (var H in homographies)
                Console.WriteLine(H);

            throw new Exception($"V is rank-deficient: last singular values are {S[4]} and {S[5]}");
        }

        VectorD b = svd.VT.Row(5);

        if (b.Any(x => double.IsNaN(x) || double.IsInfinity(x)))
            throw new Exception("SVD produced invalid solution");

        double B11 = b[0];
        double B12 = b[1];
        double B22 = b[2];
        double B13 = b[3];
        double B23 = b[4];
        double B33 = b[5];

        double v0Numer = B12 * B13 - B11 * B23;
        double v0Denom = B11 * B22 - B12 * B12;

        if (Math.Abs(v0Denom) < 1e-14 || Math.Abs(B11) < 1e-14)
            throw new Exception("Degenerate intrinsic system");

        double v0 = v0Numer / v0Denom;
        double lambda = B33 - (B13 * B13 + v0 * v0Numer) / B11;

        if (Math.Abs(lambda) < 1e-14)
            throw new Exception("Degenerate intrinsic system");

        double alpha = Math.Sqrt(lambda / B11);
        double beta = Math.Sqrt(lambda * B11 / v0Denom);

        double gamma = -B12 * alpha * alpha * beta / lambda;

        double u0 = gamma * v0 / beta - B13 * alpha * alpha / lambda;

        return DenseMatrixD.OfArray(new double[,]
        {
            { alpha, gamma, u0 },
            { 0, beta, v0 },
            { 0, 0, 1 }
        });
    }

    static VectorD Cross(
            VectorD a,
            VectorD b)
    {
        return DenseVectorD.OfArray(new[]
        {
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        });
    }

    public static void ComputeExtrinsics(MatrixD H, MatrixD Kinv, out MatrixD R, out VectorD t)
    {
        var B = Kinv * H;

        var b1 = B.Column(0);
        var b2 = B.Column(1);
        var b3 = B.Column(2);

        double lambda = 2.0 / (b1.L2Norm() + b2.L2Norm());

        var r1 = lambda * b1;
        var r2 = lambda * b2;
        t = lambda * b3;

        var r3 = Cross(r1, r2);

        R = DenseMatrixD.OfColumnVectors(r1, r2, r3);

        var svd = R.Svd(true);
        var U = svd.U;
        var Vt = svd.VT;

        var D = DenseMatrixD.CreateIdentity(3);

        if ((U * Vt).Determinant() < 0)
            D[2, 2] = -1;

        R = U * D * Vt;
    }

    public static void ComputeExtrinsics(List<MatrixD> homographies, MatrixD K, out List<MatrixD> Rs, out List<VectorD> ts)
    {
        var Kinv = K.Inverse();

        Rs = new List<MatrixD>();
        ts = new List<VectorD>();

        foreach (var H in homographies)
        {
            ComputeExtrinsics(H, Kinv, out MatrixD R, out VectorD t);

            Rs.Add(R);
            ts.Add(t);
        }
    }

    public static List<VectorD> TransformPoints(List<VectorD> points, MatrixD R, VectorD t)
    {
        List<VectorD> transformedPoints = new List<VectorD>();

        foreach (VectorD point in points)
        {
            transformedPoints.Add(R * point + t);
        }

        return transformedPoints;
    }

    public static List<List<VectorD>> TransformPoints(List<List<VectorD>> points, List<MatrixD> Rs, List<VectorD> ts)
    {
        List<List<VectorD>> transformedPoints = new List<List<VectorD>>();

        if (points.Count != Rs.Count || points.Count != ts.Count) return transformedPoints;

        for (int i = 0; i < points.Count; i++)
        {
            transformedPoints.Add(TransformPoints(points[i], Rs[i], ts[i]));
        }

        return transformedPoints;
    }

    public static double ComputeError(
    List<VectorD> image1Points,
    List<VectorD> image2Points)
    {
        double error = 0;

        if (image1Points.Count != image2Points.Count)
            throw new Exception("Invalid input points");

        for (int i = 0; i < image1Points.Count; i++)
        {
            double dx = image1Points[i][0] - image2Points[i][0];
            double dy = image1Points[i][1] - image2Points[i][1];

            error += dx * dx + dy * dy;
        }

        return error / image1Points.Count;
    }

    private static void ProjectWithJacobian(VectorD P, MatrixD K, VectorD d, out VectorD proj, out MatrixD J)
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

        double xDist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
        double yDist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;

        double u = fx * xDist + cx;
        double v = fy * yDist + cy;

        proj = DenseVectorD.OfArray(new double[] { u, v });

        J = DenseMatrixD.Create(2, 5, 0);

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

    private static void ProjectWithJacobian(
        VectorD P,
        VectorD x,
        out VectorD proj,
        out MatrixD J)
    {
        double fx = x[0];
        double fy = x[1];
        double cx = x[2];
        double cy = x[3];

        double k1 = x[4];
        double k2 = x[5];
        double p1 = x[6];
        double p2 = x[7];
        double k3 = x[8];

        double xn = P[0] / P[2];
        double yn = P[1] / P[2];

        double r2 = xn * xn + yn * yn;
        double r4 = r2 * r2;
        double r6 = r4 * r2;

        double radial = 1 + k1 * r2 + k2 * r4 + k3 * r6;

        double xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn);
        double yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn;

        double u = fx * xd + cx;
        double v = fy * yd + cy;

        proj = DenseVectorD.OfArray([u, v]);

        // 2 residuals x 9 parameters
        J = DenseMatrixD.Create(2, 9, 0);

        // fx fy cx cy

        J[0, 0] = xd;
        J[1, 0] = 0;

        J[0, 1] = 0;
        J[1, 1] = yd;

        J[0, 2] = 1;
        J[1, 2] = 0;

        J[0, 3] = 0;
        J[1, 3] = 1;

        // distortion derivatives
        J[0, 4] = fx * xn * r2;
        J[1, 4] = fy * yn * r2;

        J[0, 5] = fx * xn * r4;
        J[1, 5] = fy * yn * r4;

        J[0, 6] = fx * (2 * xn * yn);
        J[1, 6] = fy * (r2 + 2 * yn * yn);

        J[0, 7] = fx * (r2 + 2 * xn * xn);
        J[1, 7] = fy * (2 * xn * yn);

        J[0, 8] = fx * xn * r6;
        J[1, 8] = fy * yn * r6;
    }

    private static void ComputeResidualAndJacobian(
        List<VectorD> world,
        List<VectorD> image,
        MatrixD K,
        VectorD d,
        out DenseVectorD r,
        out DenseMatrixD J)
    {
        int n = world.Count;

        r = DenseVectorD.Create(2 * n, 0);
        J = DenseMatrixD.Create(2 * n, d.Count, 0);

        for (int i = 0; i < n; i++)
        {
            ProjectWithJacobian(world[i], K, d, out VectorD proj, out MatrixD jac);

            double dx = proj[0] - image[i][0];
            double dy = proj[1] - image[i][1];

            r[2 * i + 0] = dx;
            r[2 * i + 1] = dy;

            for (int k = 0; k < d.Count; k++)
            {
                J[2 * i + 0, k] = jac[0, k];
                J[2 * i + 1, k] = jac[1, k];
            }
        }
    }

    private static void ComputeResidualAndJacobian(
        List<VectorD> world,
        List<VectorD> image,
        VectorD parameters,
        out DenseVectorD r,
        out DenseMatrixD J)
    {
        int n = world.Count;

        r = DenseVectorD.Create(2 * n, 0);
        J = DenseMatrixD.Create(2 * n, 9, 0);

        for (int i = 0; i < n; i++)
        {
            ProjectWithJacobian(world[i], parameters, out VectorD proj, out MatrixD jac);

            r[2 * i + 0] = proj[0] - image[i][0];
            r[2 * i + 1] = proj[1] - image[i][1];

            for (int k = 0; k < 9; k++)
            {
                J[2 * i + 0, k] = jac[0, k];
                J[2 * i + 1, k] = jac[1, k];
            }
        }
    }

    public static void LevenbergMarquardt(
        List<VectorD> world,
        List<VectorD> image,
        MatrixD K,
        ref VectorD distortion,
        int maxIter = 30)
    {
        double lambda = 1e-3;

        for (int iter = 0; iter < maxIter; iter++)
        {
            ComputeResidualAndJacobian(world, image, K, distortion, out DenseVectorD r, out DenseMatrixD J);

            var JT = J.Transpose();
            var A = JT * J;

            var g = JT * r;

            // damping (LM)
            for (int i = 0; i < A.RowCount; i++)
                A[i, i] *= lambda;

            var h = A.Svd(true).Solve(-g);

            VectorD xNew = DenseVectorD.OfArray([0, 0, 0, 0, 0]);
            for (int i = 0; i < distortion.Count; i++)
                xNew[i] = distortion[i] + h[i];

            ComputeResidualAndJacobian(world, image, K, xNew, out DenseVectorD rNew, out DenseMatrixD _);

            double err = r.DotProduct(r);
            double errNew = rNew.DotProduct(rNew);

            if (errNew < err)
            {
                distortion = xNew;
                lambda *= 0.7;
            }
            else
            {
                lambda *= 2.0;
            }
        }
    }

    public static void LevenbergMarquardtWithK(
        List<VectorD> world,
        List<VectorD> image,
        ref MatrixD K,
        ref VectorD distortion,
        int maxIter = 50)
    {

        VectorD x =
            DenseVectorD.OfArray(
            [
                K[0,0],K[1,1],K[0,2],K[1,2],
                distortion[0],distortion[1],distortion[2],distortion[3],distortion[4]
            ]);

        double lambda = 1e-3;

        for (int iter = 0; iter < maxIter; iter++)
        {
            ComputeResidualAndJacobian(world, image, x, out var r, out var J);

            var JT = J.Transpose();

            var A = JT * J + lambda * DenseMatrixD.CreateIdentity(9);
            var g = JT * r;
            var dx = A.Svd(true).Solve(-g);
            var xNew = x + dx;

            ComputeResidualAndJacobian(world, image, xNew, out var rNew, out _);

            if (rNew.DotProduct(rNew) < r.DotProduct(r))
            {
                x = xNew;
                lambda *= 0.7;
            }
            else
            {
                lambda *= 2;
            }

            if (dx.L2Norm() < 1e-10)
                break;
        }

        K = DenseMatrixD.OfArray(new double[,]
                {
                    {x[0],0,x[2]},
                    {0,x[1],x[3]},
                    {0,0,1}
                });

        distortion = DenseVectorD.OfArray([x[4], x[5], x[6], x[7], x[8]]);
    }

    private static MatrixD EnforceEssential(
        MatrixD E)
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

        if (svd.S.Count == 9)
        {
            Console.WriteLine("svd.S");
            Console.WriteLine(svd.S);
        }

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
            U = U * DenseMatrixD.OfDiagonalArray(
                [1, 1, -1]);

        if (Vt.Determinant() < 0)
            Vt = Vt * DenseMatrixD.OfDiagonalArray(
                [1, 1, -1]);

        var W = DenseMatrixD.OfArray(new double[,] {
            {0,-1,0},
            {1, 0,0},
            {0, 0,1}
        });

        MatrixD R1 = U * W * Vt;
        MatrixD R2 = U * W.Transpose() * Vt;

        if (R1.Determinant() < 0)
            R1 = -R1;

        if (R2.Determinant() < 0)
            R2 = -R2;

        VectorD t = U.Column(2);

        Console.WriteLine("R1");
        Console.WriteLine(R1);
        Console.WriteLine("R2");
        Console.WriteLine(R2);

        Console.WriteLine("t1");
        Console.WriteLine(t);
        Console.WriteLine("t2");
        Console.WriteLine(-t);

        VectorD cameraOffset1 = R1.Transpose() * t;
        VectorD cameraOffset2 = R1.Transpose() * -t;
        VectorD cameraOffset3 = R2.Transpose() * t;
        VectorD cameraOffset4 = R2.Transpose() * -t;
        Console.WriteLine("c1");
        Console.WriteLine(cameraOffset1);
        Console.WriteLine("c2");
        Console.WriteLine(cameraOffset2);
        Console.WriteLine("c3");
        Console.WriteLine(cameraOffset3);
        Console.WriteLine("c4");
        Console.WriteLine(cameraOffset4);

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

        for (int pose = 0; pose < rotations.Count; pose++)
        {
            int count = 0;

            for (int i = 0; i < p1.Count; i++)
            {
                VectorD X = triangulate(rotations[pose], translations[pose], p1[i], p2[i]);

                if (double.IsNaN(X[0]))
                    continue;

                double z1 = X[2];

                VectorD X2 = rotations[pose] * X + translations[pose];

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

        Console.WriteLine("En");
        Console.WriteLine(En);

        // Undo Hartley normalization
        E = Denormalize(En, T1, T2);

        Console.WriteLine("E");
        Console.WriteLine(E);

        E = EnforceEssential(E);

        Console.WriteLine("E Enforced");
        Console.WriteLine(E);

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
        List<VectorD> srcNorm = CVProjection.UnProjectPoints(src, K1, d1);
        List<VectorD> dstNorm = CVProjection.UnProjectPoints(dst, K2, d2);

        // Estimate essential matrix
        List<VectorD> points1 = Normalize(srcNorm, out MatrixD T1);
        List<VectorD> points2 = Normalize(dstNorm, out MatrixD T2);

        estimateCameraPose(points1, points2, T1, T2, out MatrixD E, out var Rs, out var ts);

        // Pick the pose where points are in front
        // of both cameras
        int index = SelectPose(Rs, ts, srcNorm, dstNorm);

        R = Rs[index];
        t = ts[index];

        return SampsonError(srcNorm, dstNorm, E);
    }

    public static double SampsonError(
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

            error += numer / denom;
        }

        return error;
    }

    public static double SampsonErrorInliers(
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

        return error;
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

        double dx = 0.0;
        double dy = 0.0;
        for (int i = 0; i < camera1Points.Count; i++)
        {
            dx += camera1Points[i][0] - camera2Points[i][0];
            dy += camera1Points[i][1] - camera2Points[i][1];
        }
        dx /= camera1Points.Count;
        dy /= camera1Points.Count;

        Console.WriteLine("Mean DX, DY");
        Console.WriteLine(dx);
        Console.WriteLine(dy);

        List<VectorD> camera1NormalizedCameraCoordinates = CVProjection.UnProjectPoints(camera1Points, K1, d1);
        List<VectorD> camera2NormalizedCameraCoordinates = CVProjection.UnProjectPoints(camera2Points, K2, d2);

        // Hartley Normalize
        List<VectorD> points1 = Normalize(camera1NormalizedCameraCoordinates, out MatrixD T1);
        List<VectorD> points2 = Normalize(camera2NormalizedCameraCoordinates, out MatrixD T2);

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
            MatrixD E = Denormalize(En, T1, T2);
            E = EnforceEssential(E);

            double error = SampsonErrorInliers(camera1NormalizedCameraCoordinates, camera2NormalizedCameraCoordinates, E, threshold, out List<int> inliers);

            // 3. Keep best model
            if (inliers.Count > bestInliers.Count || (inliers.Count == bestInliers.Count && error < bestError))
            {
                bestInliers = inliers;
                bestError = error;
            }
        }

        Console.WriteLine($"Sampson Error {bestError}");
        Console.WriteLine($"Inliers {bestInliers.Count} / {n}");

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
        Console.WriteLine($"Final Sampson Error: {inlierError}");
    }

    public static List<VectorD> TriangulateAll(
        List<VectorD> src,
        List<VectorD> dst,
        MatrixD K1,
        MatrixD K2,
        VectorD d1,
        VectorD d2,
        double baseline,
        out MatrixD R,
        out VectorD t)
    {
        if (src.Count != dst.Count)
            throw new ArgumentException(
                "Point lists must have same length");

        List<VectorD> normalizedPixels1 = CVProjection.UnProjectPoints(src, K1, d1);
        List<VectorD> normalizedPixels2 = CVProjection.UnProjectPoints(dst, K2, d2);

        EstimateCameraPose(src, dst, K1, K2, d1, d2, out R, out t);

        var points = new List<VectorD>();

        for (int i = 0; i < normalizedPixels1.Count; i++)
        {
            VectorD X = triangulate(R, t, normalizedPixels1[i], normalizedPixels2[i]);
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
        // Camera centers
        VectorD C1 = DenseVectorD.OfArray([0, 0, 0]);
        VectorD C2 = -R.Transpose() * t;

        // Baseline direction
        VectorD ex = (C2 - C1).Normalize(2);

        // Average optical axis
        VectorD z1 = DenseVectorD.OfArray([0, 0, 1]);
        VectorD z2 = R.Transpose() * z1;

        VectorD ez = (z1 + z2).Normalize(2);

        // Rectified y-axis
        VectorD ey = Cross(ez, ex).Normalize(2);

        // Recompute z to make orthogonal
        ez = Cross(ex, ey).Normalize(2);

        MatrixD Rrect = DenseMatrixD.OfColumnVectors(ex, ey, ez);

        // Camera rotations into rectified frame
        R1 = Rrect;

        R2 = Rrect * R.Transpose();

        if (R1.Determinant() < 0)
        {
            R1.SetColumn(2, -R1.Column(2));
        }

        if (R2.Determinant() < 0)
        {
            R2.SetColumn(2, -R2.Column(2));
        }

        // Image homographies
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
        EstimateCameraPoseRansac(src, dst, K1, K2, d1, d2, iterations, threshold, out R, out t, out _);
        StereoRectify(K1, K2, R, t, out R1, out R2, out H1, out H2);

        double adx = 0.0;
        double ady = 0.0;
        for (int i = 0; i < src.Count; i++)
        {
            var srcA = H1 * src[i];
            var dstA = H2 * dst[i];
            adx += srcA[0] - dstA[0];
            ady += srcA[1] - dstA[1];
        }
        adx /= src.Count;
        ady /= src.Count;

        Console.WriteLine("Mean DX, DY after rectify");
        Console.WriteLine(adx);
        Console.WriteLine(ady);
    }

    public static void CalculatePlaneProjection(List<VectorD> imagePoints, List<VectorD> worldPoints, MatrixD K, VectorD d, out MatrixD R, out VectorD t)
    {
        MatrixD homography = CVCamera.ComputeHomography(imagePoints, worldPoints);

        MatrixD Kinv = K.Inverse();

        CVCamera.ComputeExtrinsics(homography, Kinv, out MatrixD RO, out VectorD tO);

        R = RO;
        t = tO;
    }

};