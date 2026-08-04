namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using System.Numerics;
using System.Diagnostics;
using MathNet.Numerics.RootFinding;
using System.Reflection;

public class CVCornerDetector
{
    private static void DerivativeX<T>(
        CVImage image,
        ref CVImage outImage) where T : struct, INumber<T>
    {
        int width = image.Width;
        int height = image.Height;
        int planeSize = width * height;

        double[,] k =
        {
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
        };

        Span<T> srcSpan = image.BufferAs<T>();
        Span<double> dstSpan = outImage.BufferAs<double>();

        for (int c = 0; c < image.Channels; c++)
        {
            int channelOffset = c * planeSize;

            for (int y = 1; y < height - 1; y++)
            {
                int dstRow = channelOffset + y * width;

                for (int x = 1; x < width - 1; x++)
                {
                    double sum = 0;

                    for (int dy = -1; dy <= 1; dy++)
                    {
                        int srcRow = channelOffset + (y + dy) * width;

                        for (int dx = -1; dx <= 1; dx++)
                        {
                            sum += double.CreateChecked(srcSpan[(x + dx) + srcRow]) * k[dy + 1, dx + 1];
                        }
                    }
                    dstSpan[x + dstRow] = sum;
                }
            }
        }

    }

    public static CVImage DerivativeX(CVImage image)
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, CVDataFormat.CV_F64, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) DerivativeX<byte>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) DerivativeX<sbyte>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) DerivativeX<ushort>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) DerivativeX<short>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) DerivativeX<uint>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) DerivativeX<int>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) DerivativeX<ulong>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) DerivativeX<long>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) DerivativeX<float>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) DerivativeX<double>(image, ref outImage);

        return outImage;
    }

    private static void DerivativeY<T>(
        CVImage image,
        ref CVImage outImage) where T : struct, INumber<T>
    {
        int width = image.Width;
        int height = image.Height;
        int planeSize = width * height;

        double[,] k =
        {
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }
    };

        Span<T> srcSpan = image.BufferAs<T>();
        Span<double> dstSpan = outImage.BufferAs<double>();

        for (int c = 0; c < image.Channels; c++)
        {
            int channelOffset = c * planeSize;

            for (int y = 1; y < height - 1; y++)
            {
                int dstRow = channelOffset + y * width;

                for (int x = 1; x < width - 1; x++)
                {
                    double sum = 0;

                    for (int dy = -1; dy <= 1; dy++)
                    {
                        int srcRow = channelOffset + (y + dy) * width;

                        for (int dx = -1; dx <= 1; dx++)
                        {
                            sum +=
                                double.CreateChecked(srcSpan[(x + dx) + srcRow]) *
                                k[dy + 1, dx + 1];
                        }
                    }
                    dstSpan[x + dstRow] = sum;
                }
            }
        }
    }

    public static CVImage DerivativeY(CVImage image)
    {
        // Derivative requires signed image
        CVImage outImage = CVImage.Create(image.Width, image.Height, CVDataFormat.CV_F64, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) DerivativeY<byte>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) DerivativeY<sbyte>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) DerivativeY<ushort>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) DerivativeY<short>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) DerivativeY<uint>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) DerivativeY<int>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) DerivativeY<ulong>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) DerivativeY<long>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) DerivativeY<float>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) DerivativeY<double>(image, ref outImage);

        return outImage;
    }

    private static void sobel(
        CVImage image,
        ref CVImage gradientX,
        ref CVImage gradientY)
    {
        int width = image.Width;
        int height = image.Height;
        int planeSize = width * height;

        Span<double> bufferSpan = image.BufferAs<double>();
        Span<double> bgx = gradientX.BufferAs<double>();
        Span<double> bgy = gradientY.BufferAs<double>();

        int lanes = Vector<double>.Count;
        var oneVec = new Vector<double>(0.125);
        var twoVec = new Vector<double>(0.250);

        for (int c = 0; c < image.Channels; c++)
        {
            int channelOffset = c * planeSize;

            for (int y = 1; y < height - 1; y++)
            {
                int r0 = channelOffset + (y - 1) * width;
                int r1 = channelOffset + y * width;
                int r2 = channelOffset + (y + 1) * width;

                int x = 1;

                if (Vector.IsHardwareAccelerated)
                {
                    for (; x <= width - lanes - 1; x += lanes)
                    {
                        Vector<double> tl = new Vector<double>(bufferSpan.Slice(r0 + x - 1));
                        Vector<double> tc = new Vector<double>(bufferSpan.Slice(r0 + x));
                        Vector<double> tr = new Vector<double>(bufferSpan.Slice(r0 + x + 1));

                        Vector<double> ml = new Vector<double>(bufferSpan.Slice(r1 + x - 1));
                        Vector<double> mr = new Vector<double>(bufferSpan.Slice(r1 + x + 1));

                        Vector<double> bl = new Vector<double>(bufferSpan.Slice(r2 + x - 1));
                        Vector<double> bc = new Vector<double>(bufferSpan.Slice(r2 + x));
                        Vector<double> br = new Vector<double>(bufferSpan.Slice(r2 + x + 1));

                        Vector<double> vx = (tr - tl) * oneVec + (mr - ml) * twoVec + (br - bl) * oneVec;
                        Vector<double> vy = (bl - tl) * oneVec + (bc - tc) * twoVec + (br - tr) * oneVec;

                        vx.CopyTo(bgx.Slice(r1 + x));
                        vy.CopyTo(bgy.Slice(r1 + x));
                    }
                }

                // scalar tail
                for (; x < width - 1; x++)
                {
                    int idx = r1 + x;

                    double tl = bufferSpan[r0 + x - 1];
                    double tc = bufferSpan[r0 + x];
                    double tr = bufferSpan[r0 + x + 1];

                    double ml = bufferSpan[r1 + x - 1];
                    double mr = bufferSpan[r1 + x + 1];

                    double bl = bufferSpan[r2 + x - 1];
                    double bc = bufferSpan[r2 + x];
                    double br = bufferSpan[r2 + x + 1];

                    bgx[idx] =
                        0.125 * (tr - tl) +
                        0.250 * (mr - ml) +
                        0.125 * (br - bl);

                    bgy[idx] =
                        0.125 * (bl - tl) +
                        0.250 * (bc - tc) +
                        0.125 * (br - tr);
                }
            }
        }
    }

    public static void Sobel(CVImage image, out CVImage gradientX, out CVImage gradientY)
    {
        CVImage doubleImage = CVConvert.ConvertDataFormat(image, CVDataFormat.CV_F64);

        gradientX = CVImage.Create(image.Width, image.Height, CVDataFormat.CV_F64, image.ChannelFormats);
        gradientY = CVImage.Create(image.Width, image.Height, CVDataFormat.CV_F64, image.ChannelFormats);

        sobel(doubleImage, ref gradientX, ref gradientY);
    }

    public static List<(int x, int y, double score)> NonMaximumSuppressionPoints(
        List<(int x, int y, double score)> points,
        float radius = 1)
    {
        List<(int x, int y, double score)> result = new List<(int x, int y, double score)>();

        foreach (var p in points.OrderByDescending(v => v.score))
        {
            bool keep = true;

            foreach (var q in result)
            {
                double dx = p.x - q.x;
                double dy = p.y - q.y;

                if (Math.Abs(dx) < radius || Math.Abs(dy) < radius)
                {
                    keep = false;
                    break;
                }
            }

            if (keep)
                result.Add(p);
        }

        return result;
    }

    public static CVImage ShiTomasiStrength(CVImage image, int windowRadius = 1)
    {
        Sobel(image, out CVImage Ix, out CVImage Iy);

        // Structure tensor components
        CVImage Ixx = Ix * Ix;
        CVImage Iyy = Iy * Iy;
        CVImage Ixy = Ix * Iy;

        // Smooth tensor components
        CVImage A = CVWindowing.AverageWindow(Ixx, windowRadius);
        CVImage B = CVWindowing.AverageWindow(Ixy, windowRadius);
        CVImage C = CVWindowing.AverageWindow(Iyy, windowRadius);

        // det(M)
        CVImage structureDeterminant = (A * C) - (B * B);

        // trace(M)
        CVImage trace = A + C;

        // Eigenvalues:
        // λ = (trace ± sqrt(trace² - 4det))/2
        CVImage discriminantSquared = (trace * trace) - (structureDeterminant * 4);
        discriminantSquared = CVMax.Max(discriminantSquared, 0);

        CVImage discriminant = CVSquareRoot.SquareRoot(discriminantSquared);

        CVImage lambda1 = (trace + discriminant) / 2;
        CVImage lambda2 = (trace - discriminant) / 2;

        return CVMin.Min(lambda1, lambda2);
    }

    public static List<(int, int, double)> DetectCornerShiTomasi(CVImage image, int windowRadius = 1, double threshold = 0.9, int nonMaxSuppressionRadius = 1)
    {
        CVImage imageC = CVConvert.ConvertChannelFormat(image, CVChannelFormat.CV_Grayscale);
        imageC = ShiTomasiStrength(imageC, windowRadius);
        imageC = CVProcessing.NonMaximumSuppression(imageC, nonMaxSuppressionRadius);
        imageC = CVProcessing.Normalize(imageC, 0, 1);
        CVImage mask = imageC > threshold;

        var pixelList = CVProcessing.GetPixels(mask, imageC, 1);
        return pixelList;
    }

    public static CVImage HarrisStrength(CVImage image, double constant = 0.04, int windowRadius = 1)
    {
        Sobel(image, out CVImage Ix, out CVImage Iy);

        CVImage A = CVWindowing.AverageWindow(Ix * Ix, windowRadius);
        CVImage B = CVWindowing.AverageWindow(Ix * Iy, windowRadius);
        CVImage C = CVWindowing.AverageWindow(Iy * Iy, windowRadius);

        CVImage det = (A * C) - (B * B);
        CVImage trace = A + C;

        return det - (trace * trace * constant);
    }

    public static CVImage HarrisCornerStrength(CVImage image, double constant = 0.04, int windowRadius = 1, int suppressionRadius = 1, double threshold = 0)
    {
        CVImage harrisStrength = HarrisStrength(image, constant, windowRadius);
        CVImage suppressed = CVProcessing.NonMaximumSuppression(harrisStrength, suppressionRadius);
        CVImage mask = suppressed > threshold;

        return mask;
    }

    public static List<(int, int, double)> DetectCornerHarris(CVImage image, double constant = 0.04, int windowRadius = 1, int threshold = 1)
    {
        CVImage gray = CVConvert.ConvertChannelFormat(image, CVChannelFormat.CV_Grayscale);
        CVImage strength = HarrisStrength(image, constant, windowRadius);
        CVImage mask = strength > threshold;

        var pixelList = CVProcessing.GetPixels(mask, strength, 1);
        return NonMaximumSuppressionPoints(pixelList);
    }

    private static double harrisStrengthSingle(
            (int x, int y) point,
            CVImage IxxImage,
            CVImage IyyImage,
            CVImage IxyImage,
            double constant,
            int windowRadius)
    {
        double sxx = 0;
        double syy = 0;
        double sxy = 0;

        Span<double> ixxBuffer = IxxImage.BufferAs<double>();
        Span<double> iyyBuffer = IyyImage.BufferAs<double>();
        Span<double> ixyBuffer = IxyImage.BufferAs<double>();

        int startX = Math.Max(point.x - windowRadius, 0);
        int startY = Math.Max(point.y - windowRadius, 0);

        int endX = Math.Min(point.x + windowRadius, IxxImage.Width - 1);
        int endY = Math.Min(point.y + windowRadius, IxxImage.Height - 1);

        for (int yy = startY; yy <= endY; yy++)
        {
            for (int xx = startX; xx <= endX; xx++)
            {
                double ixx = ixxBuffer[xx + IxxImage.Width * yy];
                double iyy = iyyBuffer[xx + IyyImage.Width * yy];
                double ixy = ixyBuffer[xx + IxyImage.Width * yy];

                sxx += ixx;
                syy += iyy;
                sxy += ixy;
            }
        }

        double det = sxx * syy - sxy * sxy;
        double trace = sxx + syy;

        return det - trace * trace * constant;
    }

    public static List<(int, int, double)> HarrisStrengthPoints(CVImage image, List<(int x, int y)> points, double constant = 0.04, int windowRadius = 1)
    {
        List<(int x, int y, double score)> scores = new List<(int x, int y, double score)>();

        if (points.Count == 0) return scores;

        Sobel(image, out CVImage gradientX, out CVImage gradientY);
        CVImage IxxImage = gradientX * gradientX;
        CVImage IyyImage = gradientY * gradientY;
        CVImage IxyImage = gradientX * gradientY;

        double harrisResponse = harrisStrengthSingle(points[0], IxxImage, IyyImage, IxyImage, constant, windowRadius);
        scores.Add((points[0].x, points[0].y, harrisResponse));
        double maxResponse = harrisResponse;

        for (int i = 1; i < points.Count; i++)
        {
            harrisResponse = harrisStrengthSingle(points[i], IxxImage, IyyImage, IxyImage, constant, windowRadius);

            if (harrisResponse <= 0) continue;
            if (harrisResponse > maxResponse) maxResponse = harrisResponse;

            scores.Add((points[i].x, points[i].y, harrisResponse));
        }

        for (int i = 0; i < scores.Count; i++)
        {
            double bound = scores[i].score / maxResponse;
            scores[i] = (scores[i].x, scores[i].y, bound);
        }

        return scores;
    }

    private static uint fastCompare<T>(T centerVal, T otherVal, T threshold) where T : INumber<T>
    {
        if (otherVal >= centerVal)
            return otherVal - centerVal > threshold ? 2u : 1u;
        else
            return centerVal - otherVal > threshold ? 0u : 1u;
    }

    private static bool fastCheckConsecutive(uint bitMask)
    {
        for (int i = 0; i < 16; i++)
        {
            uint test = (bitMask >> i) & 0x1FF; // 9 bits
            if (test == 0x1FF)
                return true;
        }

        return false;
    }
    private static int[] xOffsets = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
    private static int[] yOffsets = [-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3,];

    private static int[] earlyChecks = [1, 5, 9, 13];
    private static int[] lateChecks = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15];

    private static void fastStrength<T>(CVImage image, double threshold, int edgeDistance, ref CVImage outImage) where T : struct, INumber<T>
    {
        T thresholdT = T.CreateChecked(threshold);

        Span<T> buffer = image.BufferAs<T>();
        Span<T> dstBuffer = outImage.BufferAs<T>();

        int srcPlaneSize = image.Width * image.Height;

        for (int c = 0; c < image.Channels; c++)
        {
            int channelOffset = c * srcPlaneSize;
            for (int y = edgeDistance; y < image.Height - edgeDistance; y++)
            {
                int row = channelOffset + y * image.Width;
                for (int x = edgeDistance; x < image.Width - edgeDistance; x++)
                {
                    int center = row + x;
                    T centerValue = buffer[center];

                    // Bit masks
                    uint darkMask = 0;
                    uint brightMask = 0;

                    // Counter
                    int darkCount = 0;
                    int brightCount = 0;

                    for (int r = 0; r < earlyChecks.Length; r++)
                    {
                        int cX = x + xOffsets[earlyChecks[r]];
                        int cY = y + yOffsets[earlyChecks[r]];
                        int cRow = channelOffset + cY * image.Width;
                        uint state = fastCompare<T>(centerValue, buffer[cRow + cX], thresholdT);

                        if (state == 0)
                        {
                            darkCount++;
                            darkMask |= (1u << earlyChecks[r]);
                        }
                        else if (state == 2)
                        {
                            brightCount++;
                            brightMask |= (1u << earlyChecks[r]);
                        }
                    }

                    // Early rejection
                    if (darkCount < 3 && brightCount < 3) continue;

                    for (int r = 0; r < lateChecks.Length; r++)
                    {
                        int cX = x + xOffsets[lateChecks[r]];
                        int cY = y + yOffsets[lateChecks[r]];
                        int cRow = channelOffset + cY * image.Width;
                        uint state = fastCompare<T>(centerValue, buffer[cRow + cX], thresholdT);

                        if (state == 0)
                        {
                            darkCount++;
                            darkMask |= (1u << lateChecks[r]);
                        }
                        else if (state == 2)
                        {
                            brightCount++;
                            brightMask |= (1u << lateChecks[r]);
                        }
                    }

                    // Late-Early rejection
                    if (darkCount < 9 && brightCount < 9) continue;

                    darkMask = darkMask | (darkMask << 16);
                    brightMask = brightMask | (brightMask << 16);

                    if (fastCheckConsecutive(darkMask) || fastCheckConsecutive(brightMask))
                    {
                        dstBuffer[center] = T.One;
                    }
                }
            }
        }
    }

    public static CVImage FastStrength(CVImage image, double threshold = 5.0, int edgeDistance = 16)
    {
        CVImage stength = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) fastStrength<byte>(image, threshold, edgeDistance, ref stength);
        else if (image.DataFormat == CVDataFormat.CV_S8) fastStrength<sbyte>(image, threshold, edgeDistance, ref stength);
        else if (image.DataFormat == CVDataFormat.CV_U16) fastStrength<ushort>(image, threshold, edgeDistance, ref stength);
        else if (image.DataFormat == CVDataFormat.CV_S16) fastStrength<short>(image, threshold, edgeDistance, ref stength);
        else if (image.DataFormat == CVDataFormat.CV_U32) fastStrength<uint>(image, threshold, edgeDistance, ref stength);
        else if (image.DataFormat == CVDataFormat.CV_S32) fastStrength<int>(image, threshold, edgeDistance, ref stength);
        else if (image.DataFormat == CVDataFormat.CV_U64) fastStrength<ulong>(image, threshold, edgeDistance, ref stength);
        else if (image.DataFormat == CVDataFormat.CV_S64) fastStrength<long>(image, threshold, edgeDistance, ref stength);
        else if (image.DataFormat == CVDataFormat.CV_F32) fastStrength<float>(image, threshold, edgeDistance, ref stength);
        else if (image.DataFormat == CVDataFormat.CV_F64) fastStrength<double>(image, threshold, edgeDistance, ref stength);

        return stength;
    }

    public static CVImage CheckerboardStrength(CVImage image, int windowRadius = 1)
    {
        // First derivatives
        Sobel(image, out CVImage Ix, out CVImage Iy);

        // Structure tensor
        CVImage Ixx = Ix * Ix;
        CVImage Iyy = Iy * Iy;
        CVImage Ixy = Ix * Iy;

        CVImage A = CVWindowing.AverageWindow(Ixx, windowRadius);
        CVImage B = CVWindowing.AverageWindow(Ixy, windowRadius);
        CVImage C = CVWindowing.AverageWindow(Iyy, windowRadius);

        // Shi-Tomasi eigenvalues
        CVImage det = A * C - B * B;
        CVImage trace = A + C;

        CVImage disc = CVSquareRoot.SquareRoot(CVMax.Max(trace * trace - 4 * det, 0));

        CVImage lambda1 = (trace + disc) / 2;
        CVImage lambda2 = (trace - disc) / 2;

        // Mixed derivative
        Sobel(Ix, out CVImage Ixx2, out CVImage Ixy1);
        Sobel(Iy, out CVImage Ixy2, out CVImage Iyy2);

        CVImage IxySecond = (Ixy1 + Ixy2) / 2;

        // Checkerboard response
        CVImage isotropy = lambda2 / (lambda1 + 1e-6);

        return CVAbs.Abs(IxySecond) * isotropy;
    }

    public static List<(int, int, double)> DetectCornerCheckerboard(CVImage image, int windowRadius = 1, double threshold = 0.9, int nonMaxSuppressionRadius = 1)
    {
        CVImage imageC = CVConvert.ConvertChannelFormat(image, CVChannelFormat.CV_Grayscale);
        imageC = CheckerboardStrength(imageC, windowRadius);
        imageC = CVProcessing.NonMaximumSuppression(imageC, nonMaxSuppressionRadius);
        imageC = CVProcessing.Normalize(imageC, 0, 1);
        CVImage mask = imageC > threshold;

        var pixelList = CVProcessing.GetPixels(mask, imageC, 1);
        return pixelList;
    }
}