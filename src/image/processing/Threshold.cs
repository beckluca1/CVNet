using System.Numerics;

namespace CVNet;

public class CVThreshold
{
    public static CVImage Threshold<T>(CVImage imageIn, T threshold) where T : struct, INumber<T>
    {
        return CVBigger.Bigger(imageIn, threshold);
    }

    public static CVImage AdaptiveThresholdMean<T>(CVImage image, T offset, int radius) where T : struct, INumber<T>
    {
        CVImage blur = CVBlur.MeanBlur(image, radius);
        // Prevent from underflow / wrap around
        blur = CVMax.Max(blur, offset);
        blur = CVSubtract.Subtract(blur, offset);
        return CVSmaller.Smaller(image, blur);
    }

    public static CVImage AdaptiveThresholdGauss<T>(CVImage image, T offset, int radius) where T : struct, INumber<T>
    {
        CVImage blur = CVBlur.GaussianBlur(image, radius);
        // Prevent from underflow / wrap around
        blur = CVMax.Max(blur, offset);
        blur = CVSubtract.Subtract(blur, offset);
        return CVSmaller.Smaller(image, blur);
    }

    private static int[] otsuBucket(CVImage imageIn, CVImage histogram)
    {
        Span<uint> hBuffer = histogram.BufferAs<uint>();

        int[] thresholds = new int[imageIn.Channels];

        int bucketCount = histogram.Width;

        for (int c = 0; c < imageIn.Channels; c++)
        {
            double sumBackground = 0;
            int weightBackground = 0;
            int weightForeground = 0;

            double maxVariance = 0;

            double sum = 0;
            for (int i = 0; i < bucketCount; i++)
                sum += i * hBuffer[i];

            for (int t = 0; t < bucketCount; t++)
            {
                weightBackground += (int)hBuffer[t];
                if (weightBackground == 0)
                    continue;

                weightForeground = imageIn.Width * imageIn.Height - weightBackground;
                if (weightForeground == 0)
                    break;

                sumBackground += t * hBuffer[t];

                double meanBackground = sumBackground / weightBackground;
                double meanForeground = (sum - sumBackground) / weightForeground;

                double betweenVariance =
                    weightBackground *
                    weightForeground *
                    Math.Pow(meanBackground - meanForeground, 2);

                if (betweenVariance > maxVariance)
                {
                    maxVariance = betweenVariance;
                    thresholds[c] = t;
                }
            }
        }

        return thresholds;
    }

    private static void otsuThreshold<T>(CVImage image, int bucketCount, ref CVImage outImage) where T : struct, INumber<T>
    {
        CVImage histogram = CVProcessing.Histogram(image, bucketCount, out double min, out double max, out double bucketSize);
        int[] buckets = otsuBucket(image, histogram);
        double[] otsuThresholds = new double[buckets.Length];

        for (int i = 0; i < buckets.Length; i++)
            otsuThresholds[i] = min + bucketSize * buckets[i] - 1.0;

        outImage = CVBigger.Bigger(image, otsuThresholds);
    }

    public static CVImage OtsuThreshold(CVImage image1, int bucketCount)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) otsuThreshold<byte>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) otsuThreshold<sbyte>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) otsuThreshold<ushort>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) otsuThreshold<short>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) otsuThreshold<uint>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) otsuThreshold<int>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) otsuThreshold<ulong>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) otsuThreshold<long>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) otsuThreshold<float>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) otsuThreshold<double>(image1, bucketCount, ref outImage);

        return outImage;
    }
}