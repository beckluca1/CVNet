using System.Numerics;

namespace CVNet;

public static class CVSample
{
    private static void sampleCircular<T>(CVImage imageIn, int sampleX, int sampleY, int radius, ref List<double> sampleOut) where T : struct, INumber<T>
    {
        Span<T> buffer = imageIn.BufferAs<T>();

        int rr = radius * radius;

        for (int x = sampleX - radius; x <= sampleX + radius; x++)
        {
            if (x < 0 || x >= imageIn.Width) continue;

            int dx = x - sampleX;
            int xx = dx * dx;

            for (int y = sampleY - radius; y <= sampleY + radius; y++)
            {
                if (y < 0 || y >= imageIn.Height) continue;

                int dy = y - sampleY;
                int yy = dy * dy;

                if (xx + yy > rr) continue;

                sampleOut.Add(double.CreateChecked(buffer[x + y * imageIn.Width]));
            }
        }

        sampleOut.Sort();
    }


    public static List<double> SampleCircular(CVImage image, int sampleX, int sampleY, int radius)
    {
        int size = 2 * radius + 1;
        List<double> sampleOut = new List<double>(size * size);

        if (image.DataFormat == CVDataFormat.CV_U8) sampleCircular<byte>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) sampleCircular<sbyte>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) sampleCircular<ushort>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) sampleCircular<short>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) sampleCircular<uint>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) sampleCircular<int>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) sampleCircular<ulong>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) sampleCircular<long>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) sampleCircular<float>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) sampleCircular<double>(image, sampleX, sampleY, radius, ref sampleOut);

        return sampleOut;
    }

    public static double MedianCircular(CVImage image, int sampleX, int sampleY, int radius)
    {
        List<double> samples = SampleCircular(image, sampleX, sampleY, radius);

        if (samples.Count == 0) throw new Exception("No samples exist");

        return samples[samples.Count / 2];
    }

    public static double FilteredMedian(List<double> samples, double min, double max)
    {
        int indexMin = 0;
        int highIndexMin = samples.Count;

        while (indexMin < highIndexMin)
        {
            int mid = indexMin + (highIndexMin - indexMin) / 2;

            if (samples[mid] < min)
                indexMin = mid + 1;
            else
                highIndexMin = mid;
        }

        int indexMax = 0;
        int highIndexMax = samples.Count;

        while (indexMax < highIndexMax)
        {
            int mid = indexMax + (highIndexMax - indexMax) / 2;

            if (samples[mid] <= max)
                indexMax = mid + 1;
            else
                highIndexMax = mid;
        }

        int count = indexMax - indexMin;
        if (count <= 0) throw new Exception("No samples exist");

        return samples[indexMin + count / 2];
    }

    public static double FilteredMedianCircular(CVImage image, int sampleX, int sampleY, int radius, double min, double max)
    {
        List<double> samples = SampleCircular(image, sampleX, sampleY, radius);

        return FilteredMedian(samples, min, max);
    }

    private static void sampleSquare<T>(CVImage imageIn, int sampleX, int sampleY, int radius, ref List<double> sampleOut) where T : struct, INumber<T>
    {
        Span<T> buffer = imageIn.BufferAs<T>();

        for (int x = sampleX - radius; x <= sampleX + radius; x++)
        {
            if (x < 0 || x >= imageIn.Width) continue;

            for (int y = sampleY - radius; y <= sampleY + radius; y++)
            {
                if (y < 0 || y >= imageIn.Height) continue;

                sampleOut.Add(double.CreateChecked(buffer[x + y * imageIn.Width]));
            }
        }

        sampleOut.Sort();
    }


    public static List<double> SampleSquare(CVImage image, int sampleX, int sampleY, int radius)
    {
        int size = 2 * radius + 1;
        List<double> sampleOut = new List<double>(size * size);

        if (image.DataFormat == CVDataFormat.CV_U8) sampleSquare<byte>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) sampleSquare<sbyte>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) sampleSquare<ushort>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) sampleSquare<short>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) sampleSquare<uint>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) sampleSquare<int>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) sampleSquare<ulong>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) sampleSquare<long>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) sampleSquare<float>(image, sampleX, sampleY, radius, ref sampleOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) sampleSquare<double>(image, sampleX, sampleY, radius, ref sampleOut);

        return sampleOut;
    }

    public static double MedianSquare(CVImage image, int sampleX, int sampleY, int radius)
    {
        List<double> samples = SampleSquare(image, sampleX, sampleY, radius);

        if (samples.Count == 0) throw new Exception("No samples exist");

        return samples[samples.Count / 2];
    }

    public static double FilteredMedianSquare(CVImage image, int sampleX, int sampleY, int radius, double min, double max)
    {
        List<double> samples = SampleSquare(image, sampleX, sampleY, radius);

        return FilteredMedian(samples, min, max);
    }
}