using System.Numerics;

namespace CVNet;

public static class CVSample
{
    private static int Partition(List<double> values, int left, int right)
    {
        double pivotValue = values[right];
        int store = left;

        for (int i = left; i < right; i++)
        {
            if (values[i] < pivotValue)
            {
                (values[store], values[i]) = (values[i], values[store]);
                store++;
            }
        }

        (values[store], values[right]) = (values[right], values[store]);

        return store;
    }

    private static void QuickSelect(List<double> values, int left, int right, int k)
    {
        while (left < right)
        {
            int pivot = Partition(values, left, right);

            if (pivot == k)
                return;

            if (k < pivot)
                right = pivot - 1;
            else
                left = pivot + 1;
        }
    }

    public static double Median(List<double> samples)
    {
        if (samples.Count == 0)
            return double.NaN;

        var values = new List<double>(samples);

        int middle = values.Count / 2;

        QuickSelect(values, 0, values.Count - 1, middle);

        if (values.Count % 2 != 0)
            return values[middle];

        double upper = values[middle];

        QuickSelect(values, 0, middle - 1, middle - 1);

        return (values[middle - 1] + upper) / 2.0;
    }

    public static List<double> FilterList(List<double> samples, double min, double max)
    {
        List<double> filteredSamples = new List<double>();

        foreach (double value in samples)
        {
            if (value >= min && value <= max)
                filteredSamples.Add(value);
        }

        return filteredSamples;
    }

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

    public static double MedianAbsoluteDifference(List<double> samples, double value)
    {
        List<double> differences = new List<double>();

        foreach (double sample in samples)
            differences.Add(Math.Abs(sample - value));

        return Median(differences);
    }

    public static double MedianCircular(CVImage image, int sampleX, int sampleY, int radius)
    {
        List<double> samples = SampleCircular(image, sampleX, sampleY, radius);
        return Median(samples);
    }

    public static double MedianCircular(CVImage image, int sampleX, int sampleY, int radius, out double medianAbsoluteDeviation)
    {
        List<double> samples = SampleCircular(image, sampleX, sampleY, radius);
        double median = Median(samples);
        medianAbsoluteDeviation = MedianAbsoluteDifference(samples, median);
        return median;
    }

    public static double FilteredMedianCircular(CVImage image, int sampleX, int sampleY, int radius, double min, double max)
    {
        List<double> samples = SampleCircular(image, sampleX, sampleY, radius);
        List<double> filteredSamples = FilterList(samples, min, max);
        return Median(filteredSamples);
    }

    public static double FilteredMedianCircular(CVImage image, int sampleX, int sampleY, int radius, double min, double max, out double medianAbsoluteDeviation)
    {
        List<double> samples = SampleCircular(image, sampleX, sampleY, radius);
        List<double> filteredSamples = FilterList(samples, min, max);
        double median = Median(filteredSamples);
        medianAbsoluteDeviation = MedianAbsoluteDifference(filteredSamples, median);
        return median;
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
        return Median(samples);
    }

    public static double MedianSquare(CVImage image, int sampleX, int sampleY, int radius, out double medianAbsoluteDeviation)
    {
        List<double> samples = SampleSquare(image, sampleX, sampleY, radius);
        if (samples.Count == 0) throw new Exception("No samples exist");
        double median = Median(samples);
        medianAbsoluteDeviation = MedianAbsoluteDifference(samples, median);
        return median;
    }

    public static double FilteredMedianSquare(CVImage image, int sampleX, int sampleY, int radius, double min, double max)
    {
        List<double> samples = SampleSquare(image, sampleX, sampleY, radius);
        List<double> filteredSamples = FilterList(samples, min, max);
        return Median(filteredSamples);
    }

    public static double FilteredMedianSquare(CVImage image, int sampleX, int sampleY, int radius, double min, double max, out double medianAbsoluteDeviation)
    {
        List<double> samples = SampleSquare(image, sampleX, sampleY, radius);
        List<double> filteredSamples = FilterList(samples, min, max);
        double median = Median(filteredSamples);
        medianAbsoluteDeviation = MedianAbsoluteDifference(filteredSamples, median);
        return median;
    }
}