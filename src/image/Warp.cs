using System.Drawing.Imaging;
using System.Numerics;

namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;

public static class CVWarp
{
    public static MatrixD GetPerspectiveTransform(List<VectorD> src, List<VectorD> dst)
    {
        if (src.Count != 4 || dst.Count != 4)
            throw new ArgumentException("Need exactly 4 points");

        var A = MatrixD.Build.Dense(8, 8);
        var b = VectorD.Build.Dense(8);

        for (int i = 0; i < 4; i++)
        {
            double x = src[i][0];
            double y = src[i][1];

            double X = dst[i][0];
            double Y = dst[i][1];

            int r = i * 2;

            // X equation
            A[r, 0] = x;
            A[r, 1] = y;
            A[r, 2] = 1;
            A[r, 3] = 0;
            A[r, 4] = 0;
            A[r, 5] = 0;
            A[r, 6] = -X * x;
            A[r, 7] = -X * y;

            b[r] = X;

            // Y equation
            A[r + 1, 0] = 0;
            A[r + 1, 1] = 0;
            A[r + 1, 2] = 0;
            A[r + 1, 3] = x;
            A[r + 1, 4] = y;
            A[r + 1, 5] = 1;
            A[r + 1, 6] = -Y * x;
            A[r + 1, 7] = -Y * y;

            b[r + 1] = Y;
        }

        VectorD h = A.Solve(b);

        return MatrixD.Build.DenseOfArray(new double[,]
        {
            { h[0], h[1], h[2] },
            { h[3], h[4], h[5] },
            { h[6], h[7], 1.0  }
        });
    }

    public static VectorD TransformPoint(
        VectorD point,
        MatrixD H)
    {
        var p = DenseVectorD.OfArray(new[] { point[0], point[1], 1.0 });
        var r = H * p;

        if (Math.Abs(r[2]) < 1e-12)
            return DenseVectorD.OfArray(new double[] { 0.0, 0.0 });

        double x = r[0] / r[2];
        double y = r[1] / r[2];

        return DenseVectorD.OfArray(new double[] { x, y });
    }

    public static void WarpPerspective<T>(
        CVImage image,
        MatrixD Hinv,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T>
    {
        Span<T> src = image.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        int srcWidth = image.Width;
        int srcHeight = image.Height;
        int dstWidth = imageOut.Width;
        int dstHeight = imageOut.Height;

        int srcPlane = srcWidth * srcHeight;
        int dstPlane = dstWidth * dstHeight;

        int channels = imageOut.Channels;

        double h00 = Hinv[0, 0], h10 = Hinv[1, 0], h20 = Hinv[2, 0];
        double h01 = Hinv[0, 1], h11 = Hinv[1, 1], h21 = Hinv[2, 1];
        double h02 = Hinv[0, 2], h12 = Hinv[1, 2], h22 = Hinv[2, 2];

        const double eps = 1e-12;

        for (int c = 0; c < channels; c++)
        {
            int srcChannel = c * srcPlane;
            int dstChannel = c * dstPlane;

            for (int y = 0; y < dstHeight; y++)
            {
                int dstRow = dstChannel + y * dstWidth;

                // initial projection for x = 0
                double px = h01 * y + h02;
                double py = h11 * y + h12;
                double pw = h21 * y + h22;

                for (int x = 0; x < dstWidth; x++)
                {
                    if (pw < eps && pw > -eps)
                    {
                        dst[dstRow + x] = T.Zero;
                    }
                    else
                    {
                        double inv = 1.0 / pw;

                        int sx = (int)(px * inv + 0.5);
                        int sy = (int)(py * inv + 0.5);

                        if ((uint)sx < (uint)srcWidth &&
                            (uint)sy < (uint)srcHeight)
                        {
                            dst[dstRow + x] = src[srcChannel + sy * srcWidth + sx];
                        }
                        else
                        {
                            dst[dstRow + x] = T.Zero;
                        }
                    }

                    // incremental update (critical optimization)
                    px += h00;
                    py += h10;
                    pw += h20;
                }
            }
        }
    }

    public static CVImage WarpPerspective(CVImage image, List<VectorD> srcPoints, out MatrixD H)
    {
        double widthA = Math.Sqrt(
            Math.Pow(srcPoints[2][0] - srcPoints[3][0], 2) +
            Math.Pow(srcPoints[2][1] - srcPoints[3][1], 2));

        double widthB = Math.Sqrt(
            Math.Pow(srcPoints[1][0] - srcPoints[0][0], 2) +
            Math.Pow(srcPoints[1][1] - srcPoints[0][1], 2));

        double heightA = Math.Sqrt(
            Math.Pow(srcPoints[1][0] - srcPoints[2][0], 2) +
            Math.Pow(srcPoints[1][1] - srcPoints[2][1], 2));

        double heightB = Math.Sqrt(
            Math.Pow(srcPoints[0][0] - srcPoints[3][0], 2) +
            Math.Pow(srcPoints[0][1] - srcPoints[3][1], 2));

        double maxWidth = Math.Max(widthA, widthB);
        double maxHeight = Math.Max(heightA, heightB);

        double aspect = maxWidth / maxHeight;

        double imageSize = Math.Min(image.Width, image.Height);
        int targetWidth = (int)imageSize;
        int targetHeight = (int)imageSize;

        if (aspect < 1.0) targetWidth = (int)(imageSize * aspect);
        else targetHeight = (int)(imageSize / aspect);

        Console.WriteLine($"Width: {targetWidth}, Height: {targetHeight}");

        List<VectorD> dstPoints = new List<VectorD>();
        dstPoints.Add(DenseVectorD.OfArray(new double[] { 0.0, 0.0 }));
        dstPoints.Add(DenseVectorD.OfArray(new double[] { targetWidth - 1, 0.0 }));
        dstPoints.Add(DenseVectorD.OfArray(new double[] { targetWidth - 1, targetHeight - 1 }));
        dstPoints.Add(DenseVectorD.OfArray(new double[] { 0.0, targetHeight - 1 }));

        CVImage imageOut = CVImage.Create(targetWidth, targetHeight, image.ColorFormat, image.DataFormat, image.ChannelFormat);

        H = GetPerspectiveTransform(srcPoints, dstPoints);
        MatrixD HInv = H.Inverse();

        if (image.DataFormat == CVDataFormat.CV_U8) WarpPerspective<byte>(image, HInv, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) WarpPerspective<sbyte>(image, HInv, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) WarpPerspective<ushort>(image, HInv, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) WarpPerspective<short>(image, HInv, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) WarpPerspective<uint>(image, HInv, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) WarpPerspective<int>(image, HInv, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) WarpPerspective<ulong>(image, HInv, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) WarpPerspective<long>(image, HInv, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) WarpPerspective<float>(image, HInv, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) WarpPerspective<double>(image, HInv, ref imageOut);

        return imageOut;
    }
}