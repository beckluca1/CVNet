using System.Numerics;
using System.Runtime.InteropServices;
using MathNet.Numerics.Integration;

namespace CVNet;

public struct CVBriefDescriptor
{
    public ulong[] Descriptor;

    public CVBriefDescriptor()
    {
        Descriptor = [0, 0, 0, 0];
    }

    public bool this[int index]
    {
        get => (Descriptor[index / 64] & (1UL << (index % 64))) != 0;
        set
        {
            int word = index / 64;
            ulong mask = 1UL << (index % 64);

            if (value)
                Descriptor[word] |= mask;
            else
                Descriptor[word] &= ~mask;
        }
    }
}

public class CVFeatureDetector
{
    public static uint Compare<T>(T centerVal, T otherVal, T threshold) where T : INumber<T>
    {
        if (otherVal >= centerVal)
            return otherVal - centerVal > threshold ? 2u : 1u;
        else
            return centerVal - otherVal > threshold ? 0u : 1u;
    }

    public static bool CheckConsecutive(uint bitMask)
    {
        for (int i = 1; i < 12; i++)
            bitMask &= (bitMask >> i);

        return bitMask != 0;
    }

    private static void fast<T>(CVImage image, T threshold, ref List<(int, int)> keypoints) where T : struct, INumber<T>
    {
        int[] xOffsets = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
        int[] yOffsets = [-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3,];

        int[] earlyChecks = [1, 5, 9, 13];
        int[] lateChecks = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15];

        Span<T> buffer = image.BufferAs<T>();

        for (int x = 3; x < image.Width - 3; x++)
            for (int y = 3; y < image.Width - 3; y++)
            {
                T centerValue = buffer[x + image.Width * y];

                // Bit masks
                uint[] bitMasks = new uint[3];

                uint[] pixelVals = new uint[3];
                for (int r = 0; r < earlyChecks.Length; r++)
                {
                    uint state = Compare<T>(centerValue, buffer[(x + xOffsets[earlyChecks[r]]) + image.Width * (y + yOffsets[earlyChecks[r]])], threshold);
                    pixelVals[state]++;
                    bitMasks[state] |= (1u << earlyChecks[r]);
                }

                // Early rejection
                if (pixelVals[0] < 3 && pixelVals[2] < 3) continue;

                for (int r = 0; r < lateChecks.Length; r++)
                {
                    uint state = Compare<T>(centerValue, buffer[(x + xOffsets[lateChecks[r]]) + image.Width * (y + yOffsets[lateChecks[r]])], threshold);
                    bitMasks[state] |= (1u << lateChecks[r]);
                }

                uint darkMask = bitMasks[0] | (bitMasks[0] << 16);
                uint brightMask = bitMasks[2] | (bitMasks[2] << 16);

                if (CheckConsecutive(darkMask) || CheckConsecutive(brightMask))
                {
                    keypoints.Add((x, y));
                }
            }
    }

    public static List<(int, int)> Fast(CVImage image, double threshold)
    {
        List<(int, int)> keypoints = new List<(int, int)>();

        if (image.DataFormat == CVDataFormat.CV_U8) fast<byte>(image, (byte)threshold, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_S8) fast<sbyte>(image, (sbyte)threshold, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_U16) fast<ushort>(image, (ushort)threshold, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_S16) fast<short>(image, (short)threshold, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_U32) fast<uint>(image, (uint)threshold, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_S32) fast<int>(image, (int)threshold, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_U64) fast<ulong>(image, (ulong)threshold, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_S64) fast<long>(image, (long)threshold, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_F32) fast<float>(image, (float)threshold, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_F64) fast<double>(image, (double)threshold, ref keypoints);

        return keypoints;
    }

    private static (double, double) intensityCentroidSingle<T>(CVImage image, (int x, int y) point, int radius) where T : struct, INumber<T>
    {
        double intensitySum = 0.0;
        double xWeightedSum = 0.0;
        double yWeightedSum = 0.0;

        Span<T> buffer = image.BufferAs<T>();

        for (int yy = point.y - radius; yy <= point.y + radius; yy++)
        {
            int oY = yy - point.y;

            for (int xx = point.x - radius; xx <= point.x + radius; xx++)
            {
                int oX = xx - point.x;

                if (oX * oX + oY * oY > radius * radius) continue;

                double imageIntensity = (double)Convert.ChangeType(buffer[xx + image.Width * yy], typeof(double));
                intensitySum += imageIntensity;
                xWeightedSum += xx * imageIntensity;
                yWeightedSum += yy * imageIntensity;
            }
        }

        double centroidX = xWeightedSum / intensitySum;
        double centroidY = yWeightedSum / intensitySum;

        return (centroidX, centroidY);
    }

    private static void intensityCentroids<T>(CVImage image, List<(int, int)> points, int radius, ref List<(double, double)> centroids) where T : struct, INumber<T>
    {
        for (int i = 0; i < points.Count; i++)
        {
            centroids.Add(intensityCentroidSingle<T>(image, points[i], radius));
        }
    }

    public static List<(double, double)> IntensityCentroids(CVImage image, List<(int, int)> points, int radius)
    {
        List<(double, double)> centroids = new List<(double, double)>();

        if (image.DataFormat == CVDataFormat.CV_U8) intensityCentroids<byte>(image, points, radius, ref centroids);
        else if (image.DataFormat == CVDataFormat.CV_S8) intensityCentroids<sbyte>(image, points, radius, ref centroids);
        else if (image.DataFormat == CVDataFormat.CV_U16) intensityCentroids<ushort>(image, points, radius, ref centroids);
        else if (image.DataFormat == CVDataFormat.CV_S16) intensityCentroids<short>(image, points, radius, ref centroids);
        else if (image.DataFormat == CVDataFormat.CV_U32) intensityCentroids<uint>(image, points, radius, ref centroids);
        else if (image.DataFormat == CVDataFormat.CV_S32) intensityCentroids<int>(image, points, radius, ref centroids);
        else if (image.DataFormat == CVDataFormat.CV_U64) intensityCentroids<ulong>(image, points, radius, ref centroids);
        else if (image.DataFormat == CVDataFormat.CV_S64) intensityCentroids<long>(image, points, radius, ref centroids);
        else if (image.DataFormat == CVDataFormat.CV_F32) intensityCentroids<float>(image, points, radius, ref centroids);
        else if (image.DataFormat == CVDataFormat.CV_F64) intensityCentroids<double>(image, points, radius, ref centroids);

        return centroids;
    }

    public static List<double> Angles(List<(int x, int y)> points, List<(double x, double y)> centroids)
    {
        List<double> angles = new List<double>();

        for (int i = 0; i < points.Count; i++)
        {
            angles.Add(Math.Atan2(centroids[i].x - points[i].y, centroids[i].y - points[i].x));
        }

        return angles;
    }

    public static int[] bit_pattern_31 =
    {
        8,-3,   9, 5,
        4, 2,   7,-12,
        -11, 9,  -8, 2,
        7,-12, 12,-13,
        2,-13,  2,12,
        1,-7,   1,6,
        -2,-10, -2,-4,
        -13,-13,-11,-8,
        -13,-3, -12,-9,
        10,4,   11,9,
        -13,-8, -8,-9,
        -11,7,  -9,12,
        7,7,   12,6,
        -4,-5,  -3,0,
        -13,2,  -12,-3,
        -9,0,   -7,5,
        12,-6,   12,-1,
        -3,6,    -2,12,
        -6,-13,  -4,-8,
        11,-13,  12,-8,
        4,7,     5,1,
        5,-3,   10,-3,
        3,-7,    6,12,
        -8,-7,   -6,-2,
        -2,11,   -1,-10,
        -13,12,   -8,10,
        -7,3,    -5,-3,
        -4,2,    -3,7,
        -10,-12, -6,-7,
        -3,-5,    0,-3,
        1,-2,    4,-2,
        5,1,     8,6,
        -7,-7,   -5,-2,
        -10,-7,  -6,-3,
        3,4,     4,9,
        -4,-4,   -3,-2,
        2,6,     5,11,
        6,-10,  10,-6,
        0,4,     3,10,
        -10,-4,  -10,-9,
        2,-4,    3,-9,
        0,-4,    2,-3,
        4,-7,    6,-12,
        -10,2,   -9,-3,
        1,1,     3,4,
        2,-6,    3,-10,
        7,-3,    9,5,
        -4,-7,   -3,-12,
        9,-5,    12,-10,
        4,7,     7,12,
        6,-1,    7,1,
        8,-6,   12,-5,
        5,11,    8,9,
        -6,-7,   -3,-2,
        -2,-5,    0,-2,
        -4,-9,   -3,-5,
        -1,-6,    1,-1,
        4,-6,    6,-11,
        7,-5,    9,-4,
        5,5,     7,10,
        -8,4,    -6,8,
        -4,-5,   -2,-1,
        0,-5,    1,-10,
        -7,-2,   -5,1,
        -4,6,    -2,11,
        2,-5,    4,-9,
        6,5,     8,7,
        -6,-4,   -4,-1,
        -3,-10,  -1,-5,
        1,8,     3,12,
        5,-5,    8,-2,
        -6,1,    -4,5,
        3,-2,    5,3,
        7,-6,    10,-3,
        -5,-3,   -3,0,
        -4,8,    -1,12,
        0,6,     2,9,
        3,-7,    5,-11,
        -7,-5,   -5,-1,
        -3,2,     0,5,
        1,-1,    4,2,
        6,-2,    8,1,
        -8,-8,   -6,-4,
        -2,-8,    0,-4,
        2,7,     5,10,
        4,-10,   7,-6,
        9,-3,    11,1,
        -7,6,    -4,10,
        -2,2,     1,6,
        4,1,     7,4,
        8,-2,    10,3,
        -12,-3,  -9,0,
        -5,-6,   -3,-2,
        -1,5,     2,8,
        5,-8,    8,-5,
        9,3,    12,8,
        -10,7,   -7,11,
        -5,-10,  -2,-7,
        0,-1,    3,2
    };

    private static CVBriefDescriptor briefSingle<T>(CVImage image, (int x, int y) point, double angle) where T : struct, INumber<T>
    {
        double cosA = Math.Cos(angle);
        double sinA = Math.Sin(angle);

        Span<T> buffer = image.BufferAs<T>();

        CVBriefDescriptor briefDescriptor = new CVBriefDescriptor();
        for (int i = 0; i < 256; i++)
        {
            int offset = i * 4;

            int index0X = bit_pattern_31[offset + 0];
            int index0Y = bit_pattern_31[offset + 1];
            int index1X = bit_pattern_31[offset + 2];
            int index1Y = bit_pattern_31[offset + 3];

            int rotated0X = (int)(index0X * cosA - index0Y * sinA);
            int rotated0Y = (int)(index0X * sinA + index0Y * cosA);

            int rotated1X = (int)(index1X * cosA - index1Y * sinA);
            int rotated1Y = (int)(index1X * sinA + index1Y * cosA);

            T imageP1 = buffer[(point.x + rotated0X) + image.Width * (point.y + rotated0Y)];
            T imageP2 = buffer[(point.x + rotated1X) + image.Width * (point.y + rotated1Y)];
            briefDescriptor[i] = imageP1 < imageP2;
        }

        return briefDescriptor;
    }

    private static void brief<T>(CVImage image, List<(int x, int y)> points, List<double> angles, ref List<CVBriefDescriptor> descriptors) where T : struct, INumber<T>
    {
        for (int i = 0; i < points.Count; i++)
        {
            descriptors.Add(briefSingle<T>(image, points[i], angles[i]));
        }
    }

    public static List<CVBriefDescriptor> Brief(CVImage image, List<(int, int)> points, List<double> angles)
    {
        List<CVBriefDescriptor> descriptors = new List<CVBriefDescriptor>();

        if (image.DataFormat == CVDataFormat.CV_U8) brief<byte>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_S8) brief<sbyte>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_U16) brief<ushort>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_S16) brief<short>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_U32) brief<uint>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_S32) brief<int>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_U64) brief<ulong>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_S64) brief<long>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_F32) brief<float>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_F64) brief<double>(image, points, angles, ref descriptors);

        return descriptors;
    }

    public static List<CVBriefDescriptor> Orb(CVImage image)
    {
        CVImage gray = CVConvert.ConvertChannelFormat(image, CVChannelFormat.CV_Grayscale);
        CVImage grayC = CVConvert.ConvertDataFormatToFloat(gray);

        List<(int, int)> fastKeypoints = Fast(grayC, 20);
        List<double> fastHarrisScores = CVCornerDetector.HarrisStrengthPoints(grayC, fastKeypoints, 25, 3);

        var indices = Enumerable.Range(0, fastHarrisScores.Count)
                            .OrderByDescending(i => fastHarrisScores[i])
                            .Take(100)
                            .ToList();

        List<(int, int)> bestFastPoints = indices.Select(i => fastKeypoints[i]).ToList();
        List<(double, double)> intensityCentroids = IntensityCentroids(grayC, bestFastPoints, 31);
        List<double> angles = Angles(bestFastPoints, intensityCentroids);
        return Brief(grayC, bestFastPoints, angles);
    }

    public static List<(int, int, double)> DetectFeatureFast(
        CVImage image)
    {
        CVImage gray = CVConvert.ConvertChannelFormat(image, CVChannelFormat.CV_Grayscale);
        CVImage grayC = CVConvert.ConvertDataFormatToFloat(gray);

        List<(int, int)> fastKeypoints = Fast(grayC, 20);
        List<double> fastHarrisScores = CVCornerDetector.HarrisStrengthPoints(grayC, fastKeypoints, 25, 3);

        var indices = Enumerable.Range(0, fastHarrisScores.Count)
                            .OrderByDescending(i => fastHarrisScores[i])
                            .Take(100)
                            .ToList();

        var bestFastPoints = indices
            .Select(i => (x: fastKeypoints[i].Item1, y: fastKeypoints[i].Item2, score: fastHarrisScores[i]))
            .ToList();

        return bestFastPoints;
    }
}