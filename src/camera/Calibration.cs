namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using DenseMatrixD = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix;

public class CVCalibration
{
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

        var r3 = CVCamera.Cross(r1, r2);

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

    public static double ComputeMeanSquaredError(
    List<VectorD> points1,
    List<VectorD> points2)
    {
        double error = 0;

        if (points1.Count != points2.Count)
            throw new Exception("Invalid input points");

        for (int i = 0; i < points1.Count; i++)
        {
            double dx = points1[i][0] - points2[i][0];
            double dy = points1[i][1] - points2[i][1];

            error += dx * dx + dy * dy;
        }

        return error / points1.Count;
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
            A += lambda * DenseMatrixD.CreateIdentity(A.RowCount);
            var g = JT * r;

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

            var A = JT * J;
            A += lambda * DenseMatrixD.CreateIdentity(A.RowCount);
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

    public static double Calibrate(List<List<VectorD>> imagePointGroups, List<List<VectorD>> worldPointGroups, out MatrixD K, out VectorD d)
    {
        if (imagePointGroups.Count == 0 || imagePointGroups.Count != worldPointGroups.Count) throw new Exception("Invalid number of samples");

        List<MatrixD> homographies = new List<MatrixD>();

        for (int i = 0; i < imagePointGroups.Count; i++)
        {
            if (imagePointGroups[i].Count == 0 || imagePointGroups[i].Count != worldPointGroups[i].Count) throw new Exception("Invalid number of points");

            homographies.Add(CVHomography.ComputeHomography(worldPointGroups[i], imagePointGroups[i]));
        }

        K = ComputeIntrinsics(homographies);

        ComputeExtrinsics(homographies, K, out List<MatrixD> Rs, out List<VectorD> ts);

        d = DenseVectorD.OfArray([0, 0, 0, 0, 0]);

        List<List<VectorD>> transformedWorldPoints = TransformPoints(worldPointGroups, Rs, ts);

        List<VectorD> imagePoints = new List<VectorD>();
        List<VectorD> worldPoints = new List<VectorD>();

        for (int i = 0; i < imagePointGroups.Count; i++)
        {
            foreach (VectorD point in imagePointGroups[i])
                imagePoints.Add(point);
            foreach (VectorD point in transformedWorldPoints[i])
                worldPoints.Add(point);
        }

        LevenbergMarquardtWithK(worldPoints, imagePoints, ref K, ref d);

        List<VectorD> projectedWorldPoints = CVProjection.IntrinsicProjectPoints(worldPoints, K, d);

        return ComputeMeanSquaredError(imagePoints, projectedWorldPoints);
    }

    private static double rotationDifference(MatrixD R1, MatrixD R2)
    {
        MatrixD dR = R1 * R2.Transpose();

        double c = (dR.Trace() - 1.0) * 0.5;

        c = Math.Max(-1.0, Math.Min(1.0, c));

        return Math.Acos(c); // radians
    }

    public static void RelativeTransformations(List<MatrixD> Rs1, List<VectorD> ts1, List<MatrixD> Rs2, List<VectorD> ts2, out List<MatrixD> Rs, out List<VectorD> ts)
    {
        Rs = new List<MatrixD>();
        ts = new List<VectorD>();

        for (int i = 0; i < Rs1.Count; i++)
        {
            MatrixD R = Rs2[i] * Rs1[i].Transpose();
            VectorD t = ts2[i] - R * ts1[i];

            if (Rs.Count > 0)
            {
                double relativeRotation = rotationDifference(Rs[0], R);
                if (Math.Abs(relativeRotation) > 15.0 * Math.PI / 180.0) continue;
            }

            Console.WriteLine(R);
            Console.WriteLine(t);

            Rs.Add(R);
            ts.Add(t);
        }
    }

    private static System.Numerics.Quaternion rotationMatrixToQuaternion(MatrixD R)
    {
        var m = new System.Numerics.Matrix4x4(
            (float)R[0, 0], (float)R[0, 1], (float)R[0, 2], 0,
            (float)R[1, 0], (float)R[1, 1], (float)R[1, 2], 0,
            (float)R[2, 0], (float)R[2, 1], (float)R[2, 2], 0,
            0, 0, 0, 1);

        return System.Numerics.Quaternion.CreateFromRotationMatrix(m);
    }

    private static MatrixD quaternionToRotationMatrix(System.Numerics.Quaternion q)
    {
        var m = System.Numerics.Matrix4x4.CreateFromQuaternion(q);

        return DenseMatrixD.OfArray(new double[,]
        {
            { m.M11, m.M12, m.M13 },
            { m.M21, m.M22, m.M23 },
            { m.M31, m.M32, m.M33 }
        });
    }

    public static void AverageTransformations(List<MatrixD> Rs, List<VectorD> ts, out MatrixD R, out VectorD t)
    {
        t = DenseVectorD.Create(3, 0);

        foreach (var ti in ts)
            t += ti;

        t /= ts.Count;

        List<System.Numerics.Quaternion> quats = new();

        foreach (var Ri in Rs)
        {
            var q = rotationMatrixToQuaternion(Ri);

            // Keep all quaternions on the same hemisphere.
            if (quats.Count > 0 &&
                System.Numerics.Quaternion.Dot(quats[0], q) < 0)
            {
                q = new System.Numerics.Quaternion(
                    -q.X,
                    -q.Y,
                    -q.Z,
                    -q.W);
            }

            quats.Add(q);
        }

        float x = 0;
        float y = 0;
        float z = 0;
        float w = 0;

        foreach (var q in quats)
        {
            x += q.X;
            y += q.Y;
            z += q.Z;
            w += q.W;
        }

        var avg = new System.Numerics.Quaternion(x, y, z, w);
        avg = System.Numerics.Quaternion.Normalize(avg);

        R = quaternionToRotationMatrix(avg);
    }

    public static double StereoCalibrate(List<List<VectorD>> image1PointGroups, List<List<VectorD>> image2PointGroups, List<List<VectorD>> worldPointGroups, out MatrixD K1, out MatrixD K2, out VectorD d1, out VectorD d2, out MatrixD R, out VectorD t)
    {
        if (image1PointGroups.Count == 0 || image1PointGroups.Count != image2PointGroups.Count || image1PointGroups.Count != worldPointGroups.Count) throw new Exception("Invalid number of samples");

        List<MatrixD> homographies1 = new List<MatrixD>();
        List<MatrixD> homographies2 = new List<MatrixD>();

        for (int i = 0; i < image1PointGroups.Count; i++)
        {
            if (image1PointGroups[i].Count == 0 || image1PointGroups[i].Count != image2PointGroups[i].Count || image1PointGroups[i].Count != worldPointGroups[i].Count) throw new Exception("Invalid number of points");

            homographies1.Add(CVHomography.ComputeHomography(worldPointGroups[i], image1PointGroups[i]));
            homographies2.Add(CVHomography.ComputeHomography(worldPointGroups[i], image2PointGroups[i]));
        }

        K1 = ComputeIntrinsics(homographies1);
        K2 = ComputeIntrinsics(homographies2);

        ComputeExtrinsics(homographies1, K1, out List<MatrixD> Rs1, out List<VectorD> ts1);
        ComputeExtrinsics(homographies2, K2, out List<MatrixD> Rs2, out List<VectorD> ts2);

        RelativeTransformations(Rs1, ts1, Rs2, ts2, out List<MatrixD> Rs, out List<VectorD> ts);
        AverageTransformations(Rs, ts, out R, out t);

        d1 = DenseVectorD.OfArray([0, 0, 0, 0, 0]);
        d2 = DenseVectorD.OfArray([0, 0, 0, 0, 0]);

        List<List<VectorD>> transformedWorldPoints1 = TransformPoints(worldPointGroups, Rs1, ts1);
        List<List<VectorD>> transformedWorldPoints2 = TransformPoints(worldPointGroups, Rs2, ts2);

        List<VectorD> image1Points = new List<VectorD>();
        List<VectorD> image2Points = new List<VectorD>();
        List<VectorD> worldPoints1 = new List<VectorD>();
        List<VectorD> worldPoints2 = new List<VectorD>();

        for (int i = 0; i < image1PointGroups.Count; i++)
        {
            foreach (VectorD point in image1PointGroups[i])
                image1Points.Add(point);
            foreach (VectorD point in transformedWorldPoints1[i])
                worldPoints1.Add(point);

            foreach (VectorD point in image2PointGroups[i])
                image2Points.Add(point);
            foreach (VectorD point in transformedWorldPoints2[i])
                worldPoints2.Add(point);
        }

        LevenbergMarquardtWithK(worldPoints1, image1Points, ref K1, ref d1);
        LevenbergMarquardtWithK(worldPoints2, image2Points, ref K2, ref d2);

        List<VectorD> projectedWorldPoints1 = CVProjection.IntrinsicProjectPoints(worldPoints1, K1, d1);
        List<VectorD> projectedWorldPoints2 = CVProjection.IntrinsicProjectPoints(worldPoints2, K2, d2);

        double error1 = ComputeMeanSquaredError(image1Points, projectedWorldPoints1);
        double error2 = ComputeMeanSquaredError(image2Points, projectedWorldPoints2);

        return error1 + error2;
    }
};