using System.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using DenseMatrixD = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix;

public class CVHomographySolver
{
    private static double[] Poly1Mul(
    double[] p,
    double[] q)
    {
        // p,q:
        //
        // [x, y, z, 1]
        //
        // result:
        //
        // [x², xy, y², xz, yz, z², x, y, z, 1]

        return
        [
            p[0] * q[0],
        p[0] * q[1] + p[1] * q[0],
        p[1] * q[1],

        p[0] * q[2] + p[2] * q[0],
        p[1] * q[2] + p[2] * q[1],
        p[2] * q[2],

        p[0] * q[3] + p[3] * q[0],
        p[1] * q[3] + p[3] * q[1],
        p[2] * q[3] + p[3] * q[2],

        p[3] * q[3]
        ];
    }


    private static double[] Poly2Add(
        double[] a,
        double[] b)
    {
        double[] r = new double[10];

        for (int i = 0; i < 10; i++)
            r[i] = a[i] + b[i];

        return r;
    }


    private static double[] Poly2Sub(
        double[] a,
        double[] b)
    {
        double[] r = new double[10];

        for (int i = 0; i < 10; i++)
            r[i] = a[i] - b[i];

        return r;
    }


    private static double[] Poly2Scale(
        double[] a,
        double s)
    {
        double[] r = new double[10];

        for (int i = 0; i < 10; i++)
            r[i] = a[i] * s;

        return r;
    }


    private static double[] Poly2MulPoly1(
        double[] p,
        double[] q)
    {
        // p:
        //
        // x² xy y² xz yz z² x y z 1
        //
        // q:
        //
        // x y z 1
        //
        // result:
        //
        // x³ x²y xy² y³ x²z xyz y²z xz² yz² z³
        // x² xy y² xz yz z² x y z 1

        double[] r = new double[20];

        r[0] =
            p[0] * q[0];

        r[1] =
            p[0] * q[1] +
            p[1] * q[0];

        r[2] =
            p[1] * q[1] +
            p[2] * q[0];

        r[3] =
            p[2] * q[1];

        r[4] =
            p[0] * q[2] +
            p[3] * q[0];

        r[5] =
            p[1] * q[2] +
            p[3] * q[1] +
            p[4] * q[0];

        r[6] =
            p[2] * q[2] +
            p[4] * q[1];

        r[7] =
            p[3] * q[2] +
            p[5] * q[0];

        r[8] =
            p[4] * q[2] +
            p[5] * q[1];

        r[9] =
            p[5] * q[2];

        r[10] =
            p[0] * q[3] +
            p[6] * q[0];

        r[11] =
            p[1] * q[3] +
            p[6] * q[1] +
            p[7] * q[0];

        r[12] =
            p[2] * q[3] +
            p[7] * q[1];

        r[13] =
            p[3] * q[3] +
            p[6] * q[2] +
            p[8] * q[0];

        r[14] =
            p[4] * q[3] +
            p[7] * q[2] +
            p[8] * q[1];

        r[15] =
            p[5] * q[3] +
            p[8] * q[2];

        r[16] =
            p[6] * q[3] +
            p[9] * q[0];

        r[17] =
            p[7] * q[3] +
            p[9] * q[1];

        r[18] =
            p[8] * q[3] +
            p[9] * q[2];

        r[19] =
            p[9] * q[3];

        return r;
    }

    private static List<MatrixD> EstimateEssential5Point(
    IReadOnlyList<VectorD> p1,
    IReadOnlyList<VectorD> p2)
    {
        if (p1.Count != 5 || p2.Count != 5)
            throw new ArgumentException(
                "The 5-point solver requires exactly 5 correspondences.");

        // ------------------------------------------------------------
        // 1. Build the 5x9 epipolar constraint matrix
        //
        // x2^T E x1 = 0
        //
        // E is stored row-major:
        //
        // e00 e01 e02 e10 e11 e12 e20 e21 e22
        // ------------------------------------------------------------

        MatrixD A = DenseMatrixD.Create(5, 9, 0.0);

        for (int i = 0; i < 5; i++)
        {
            double x1 = p1[i][0] / p1[i][2];
            double y1 = p1[i][1] / p1[i][2];

            double x2 = p2[i][0] / p2[i][2];
            double y2 = p2[i][1] / p2[i][2];

            A[i, 0] = x2 * x1;
            A[i, 1] = x2 * y1;
            A[i, 2] = x2;

            A[i, 3] = y2 * x1;
            A[i, 4] = y2 * y1;
            A[i, 5] = y2;

            A[i, 6] = x1;
            A[i, 7] = y1;
            A[i, 8] = 1.0;
        }

        // ------------------------------------------------------------
        // 2. Four-dimensional null space
        //
        // E = x*E0 + y*E1 + z*E2 + E3
        //
        // OpenCV's SVD path takes VT rows 5,6,7,8.
        // ------------------------------------------------------------

        var svd = A.Svd(true);

        MatrixD VT = svd.VT;

        VectorD E0 = VT.Row(5);
        VectorD E1 = VT.Row(6);
        VectorD E2 = VT.Row(7);
        VectorD E3 = VT.Row(8);

        // ------------------------------------------------------------
        // 3. Arrange the four null vectors as polynomial linear forms.
        //
        // nullSpaceMat[row][col] contains:
        //
        // [coefficient_x,
        //  coefficient_y,
        //  coefficient_z,
        //  constant]
        //
        // for E[row,col].
        // ------------------------------------------------------------

        double[][][] e = new double[3][][];

        e[0] =
        [
            [E0[0], E1[0], E2[0], E3[0]],
        [E0[1], E1[1], E2[1], E3[1]],
        [E0[2], E1[2], E2[2], E3[2]]
        ];

        e[1] =
        [
            [E0[3], E1[3], E2[3], E3[3]],
        [E0[4], E1[4], E2[4], E3[4]],
        [E0[5], E1[5], E2[5], E3[5]]
        ];

        e[2] =
        [
            [E0[6], E1[6], E2[6], E3[6]],
        [E0[7], E1[7], E2[7], E3[7]],
        [E0[8], E1[8], E2[8], E3[8]]
        ];

        // ------------------------------------------------------------
        // 4. Compute E * E^T as quadratic polynomials.
        //
        // eet[i][j] = sum_k E[i,k] E[j,k]
        // ------------------------------------------------------------

        double[][][] eet = new double[3][][];

        for (int i = 0; i < 3; i++)
        {
            eet[i] = new double[3][];

            for (int j = 0; j < 3; j++)
            {
                double[] q =
                    Poly1Mul(e[i][0], e[j][0]);

                q = Poly2Add(
                    q,
                    Poly1Mul(e[i][1], e[j][1]));

                q = Poly2Add(
                    q,
                    Poly1Mul(e[i][2], e[j][2]));

                eet[i][j] = q;
            }
        }

        // ------------------------------------------------------------
        // 5. Essential matrix constraints:
        //
        // E E^T E - 1/2 trace(E E^T) E = 0
        //
        // plus det(E) = 0.
        //
        // 10 equations x 20 monomials.
        // ------------------------------------------------------------

        MatrixD constraints =
            DenseMatrixD.Create(10, 20, 0.0);

        double[] trace =
            Poly2Scale(
                Poly2Add(
                    Poly2Add(
                        eet[0][0],
                        eet[1][1]),
                    eet[2][2]),
                0.5);

        // Nine cubic equations.
        //
        // OpenCV ordering:
        //
        // row = 1 + 3*j + i
        //
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double[] cubic =
                    Poly2MulPoly1(
                        eet[i][0],
                        e[0][j]);

                cubic = Add20(
                    cubic,
                    Poly2MulPoly1(
                        eet[i][1],
                        e[1][j]));

                cubic = Add20(
                    cubic,
                    Poly2MulPoly1(
                        eet[i][2],
                        e[2][j]));

                double[] correction =
                    Poly2MulPoly1(
                        trace,
                        e[i][j]);

                cubic = Sub20(
                    cubic,
                    correction);

                for (int k = 0; k < 20; k++)
                    constraints[
                        1 + 3 * j + i,
                        k] = cubic[k];
            }
        }

        // ------------------------------------------------------------
        // determinant(E) = 0
        //
        // det(E) =
        //
        // E01 E12 E20
        // + E02 E10 E21
        // + E00 E11 E22
        // - E02 E11 E20
        // - E01 E10 E22
        // - E00 E12 E21
        // ------------------------------------------------------------

        double[] m00 =
            Poly2Sub(
                Poly1Mul(e[0][1], e[1][2]),
                Poly1Mul(e[0][2], e[1][1]));

        double[] m01 =
            Poly2Sub(
                Poly1Mul(e[0][2], e[1][0]),
                Poly1Mul(e[0][0], e[1][2]));

        double[] m02 =
            Poly2Sub(
                Poly1Mul(e[0][0], e[1][1]),
                Poly1Mul(e[0][1], e[1][0]));

        double[] determinant =
            Poly2MulPoly1(m00, e[2][0]);

        determinant = Add20(
            determinant,
            Poly2MulPoly1(m01, e[2][1]));

        determinant = Add20(
            determinant,
            Poly2MulPoly1(m02, e[2][2]));

        for (int k = 0; k < 20; k++)
            constraints[0, k] = determinant[k];

        // ------------------------------------------------------------
        // 6. Compute Groebner elimination.
        //
        // OpenCV:
        //
        // Acoef = constraints[0..9,0..9]
        // Bcoef = constraints[0..9,10..19]
        //
        // eliminated = Acoef^-1 Bcoef
        // ------------------------------------------------------------

        MatrixD Acoef =
            constraints.SubMatrix(0, 10, 0, 10);

        MatrixD Bcoef =
            constraints.SubMatrix(0, 10, 10, 10);

        // Avoid explicit determinant tests because they are
        // numerically less useful than the solve itself.
        MatrixD eliminated;

        try
        {
            eliminated = Acoef.Solve(Bcoef);
        }
        catch
        {
            return new List<MatrixD>();
        }

        // Reject very ill-conditioned minimal samples.
        if (!double.IsFinite(eliminated.L2Norm()))
            return new List<MatrixD>();

        // ------------------------------------------------------------
        // 7. Construct the 10x10 action matrix.
        //
        // This is exactly the row extraction used by OpenCV's
        // Stewénius implementation.
        // ------------------------------------------------------------

        MatrixD action =
            DenseMatrixD.Create(10, 10, 0.0);

        // rows 0..2 <- eliminated rows 0..2
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 10; c++)
                action[r, c] = eliminated[r, c];
        }

        // rows 3..4 <- eliminated rows 4..5
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 10; c++)
                action[3 + r, c] =
                    eliminated[4 + r, c];
        }

        // row 5 <- eliminated row 7
        for (int c = 0; c < 10; c++)
            action[5, c] =
                eliminated[7, c];

        // Remaining monomial relations.
        action[6, 0] = -1.0;
        action[7, 1] = -1.0;
        action[8, 3] = -1.0;
        action[9, 6] = -1.0;

        // ------------------------------------------------------------
        // 8. OpenCV asks for LEFT eigenvectors.
        //
        // MathNet EVD returns RIGHT eigenvectors.
        //
        // Therefore solve:
        //
        // action^T v = lambda v
        //
        // ------------------------------------------------------------

        Matrix<Complex> actionT =
            action.Transpose()
                  .Map(v => new Complex(v, 0.0));

        var evd = actionT.Evd();

        List<MatrixD> models = new();

        for (int k = 0; k < 10; k++)
        {
            Complex lambda =
                evd.EigenValues[k];

            // Only real solutions.
            double scale =
                Math.Max(1.0, Math.Abs(lambda.Real));

            if (Math.Abs(lambda.Imaginary) >
                1e-8 * scale)
                continue;

            MathNet.Numerics.LinearAlgebra.Vector<Complex> v =
                evd.EigenVectors.Column(k);

            // The Stewénius quotient basis gives the
            // coordinates x,y,z in entries 6,7,8
            // and the constant in entry 9.
            //
            // OpenCV uses these components directly.
            //
            // We normalize them because E is homogeneous.
            Complex w = v[9];

            if (Complex.Abs(w) < 1e-12)
                continue;

            double x =
                (v[6] / w).Real;

            double y =
                (v[7] / w).Real;

            double z =
                (v[8] / w).Real;

            if (!double.IsFinite(x) ||
                !double.IsFinite(y) ||
                !double.IsFinite(z))
                continue;

            // --------------------------------------------------------
            // Construct E = x E0 + y E1 + z E2 + E3
            // --------------------------------------------------------

            MatrixD E =
                DenseMatrixD.Create(3, 3, 0.0);

            for (int j = 0; j < 9; j++)
            {
                double value =
                    E0[j] * x +
                    E1[j] * y +
                    E2[j] * z +
                    E3[j];

                E[j / 3, j % 3] = value;
            }

            double norm =
                E.FrobeniusNorm();

            if (!double.IsFinite(norm) ||
                norm < 1e-14)
                continue;

            E /= norm;

            // --------------------------------------------------------
            // Remove duplicate solutions.
            // --------------------------------------------------------

            bool duplicate = false;

            foreach (MatrixD existing in models)
            {
                double dPlus =
                    (E - existing).FrobeniusNorm();

                double dMinus =
                    (E + existing).FrobeniusNorm();

                if (Math.Min(dPlus, dMinus) < 1e-6)
                {
                    duplicate = true;
                    break;
                }
            }

            if (!duplicate)
                models.Add(E);
        }

        return models;
    }

    private static double[] Add20(
    double[] a,
    double[] b)
    {
        double[] r = new double[20];

        for (int i = 0; i < 20; i++)
            r[i] = a[i] + b[i];

        return r;
    }


    private static double[] Sub20(
        double[] a,
        double[] b)
    {
        double[] r = new double[20];

        for (int i = 0; i < 20; i++)
            r[i] = a[i] - b[i];

        return r;
    }

    private static double SampsonError(
    MatrixD E,
    VectorD x1,
    VectorD x2)
    {
        VectorD Ex1 = E * x1;
        VectorD Etx2 = E.Transpose() * x2;

        double r =
            x2.DotProduct(Ex1);

        double denom =
            Ex1[0] * Ex1[0] +
            Ex1[1] * Ex1[1] +
            Etx2[0] * Etx2[0] +
            Etx2[1] * Etx2[1];

        if (denom < 1e-15)
            return double.MaxValue;

        return r * r / denom;
    }

    public static bool EstimateEssentialRansac(
    List<VectorD> p1,
    List<VectorD> p2,
    int iterations,
    double threshold,
    double confidence,
    out MatrixD bestE,
    out List<int> bestInliers)
    {
        bestE =
            DenseMatrixD.CreateIdentity(3);

        bestInliers = new();

        int n = p1.Count;

        if (n != p2.Count || n < 5)
            return false;

        Random rng = new Random();

        double bestScore = double.MaxValue;

        for (int iteration = 0;
             iteration < iterations;
             iteration++)
        {
            // --------------------------------------------------------
            // Select 5 unique points
            // --------------------------------------------------------

            HashSet<int> sampleIndices = new();

            while (sampleIndices.Count < 5)
                sampleIndices.Add(rng.Next(n));

            List<VectorD> sample1 = new();
            List<VectorD> sample2 = new();

            foreach (int index in sampleIndices)
            {
                sample1.Add(p1[index]);
                sample2.Add(p2[index]);
            }

            List<MatrixD> candidates;

            try
            {
                candidates =
                    EstimateEssential5Point(
                        sample1,
                        sample2);
            }
            catch
            {
                continue;
            }

            // --------------------------------------------------------
            // Every 5-point sample can produce multiple E matrices.
            // Evaluate ALL of them.
            // --------------------------------------------------------

            foreach (MatrixD E in candidates)
            {
                List<int> inliers = new();
                double error = 0.0;

                for (int i = 0; i < n; i++)
                {
                    double e =
                        SampsonError(
                            E,
                            p1[i],
                            p2[i]);

                    if (e < threshold)
                    {
                        inliers.Add(i);
                        error += e;
                    }
                }

                if (inliers.Count == 0)
                    continue;

                // Primary score = number of inliers.
                //
                // Secondary score = mean Sampson error.
                double meanError =
                    error / inliers.Count;

                if (inliers.Count > bestInliers.Count ||
                    (inliers.Count == bestInliers.Count &&
                     meanError < bestScore))
                {
                    bestE = E;
                    bestInliers = inliers;
                    bestScore = meanError;
                }
            }

            // --------------------------------------------------------
            // Optional adaptive RANSAC termination.
            // --------------------------------------------------------

            if (bestInliers.Count >= 5)
            {
                double inlierRatio =
                    (double)bestInliers.Count / n;

                double pNoOutliers =
                    1.0 - Math.Pow(
                        inlierRatio,
                        5);

                if (pNoOutliers < 1e-15)
                    break;

                int required =
                    (int)Math.Ceiling(
                        Math.Log(1.0 - confidence) /
                        Math.Log(pNoOutliers));

                if (iteration + 1 >= required)
                    break;
            }
        }

        return bestInliers.Count >= 5;
    }
}