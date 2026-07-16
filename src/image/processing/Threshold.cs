using System.Numerics;

namespace CVNet;

public class CVThreshold
{
    public static CVImage Threshold<T>(CVImage imageIn, T threshold) where T : struct, INumber<T>
    {
        return CVBigger.Bigger(imageIn, threshold);
    }

    public static CVImage AdaptiveThresholdMean<T>(CVImage imageIn, T offset, int size) where T : struct, INumber<T>
    {
        // Requires Signed image because of subtraction
        CVImage signedImage = CVConvert.ConvertDataFormatToSigned(imageIn);
        CVImage meanBlur = CVBlur.MeanBlur(signedImage, size);
        meanBlur = CVSubtract.Subtract(meanBlur, offset);
        CVImage thresh = CVSmaller.Smaller(signedImage, meanBlur);
        return CVConvert.ConvertDataFormat(thresh, imageIn.DataFormat);
    }

    public static CVImage AdaptiveThresholdGauss<T>(CVImage imageIn, T offset, int size) where T : struct, INumber<T>
    {
        CVImage floatImage = CVConvert.ConvertDataFormat(imageIn, CVDataFormat.CV_F32);
        CVImage meanBlur = CVBlur.GaussianBlur(floatImage, size);
        meanBlur = CVSubtract.Subtract(meanBlur, offset);
        CVImage thresh = CVSmaller.Smaller(floatImage, meanBlur);
        return CVConvert.ConvertDataFormat(thresh, imageIn.DataFormat);
    }

    private static int otsuBucket(CVImage imageIn, List<int> histogram)
    {
        int bucketCount = histogram.Count;

        double sumBackground = 0;
        int weightBackground = 0;
        int weightForeground = 0;

        double maxVariance = 0;
        int threshold = 0;

        double sum = 0;
        for (int i = 0; i < bucketCount; i++)
            sum += i * histogram[i];

        for (int t = 0; t < bucketCount; t++)
        {
            weightBackground += histogram[t];
            if (weightBackground == 0)
                continue;

            weightForeground = imageIn.Width * imageIn.Height * imageIn.Channels - weightBackground;
            if (weightForeground == 0)
                break;

            sumBackground += t * histogram[t];

            double meanBackground = sumBackground / weightBackground;
            double meanForeground = (sum - sumBackground) / weightForeground;

            double betweenVariance =
                weightBackground *
                weightForeground *
                Math.Pow(meanBackground - meanForeground, 2);

            if (betweenVariance > maxVariance)
            {
                maxVariance = betweenVariance;
                threshold = t;
            }
        }

        return threshold;
    }

    private static void otsuThreshold<T>(CVImage imageIn, int bucketCount, ref CVImage outImage) where T : struct, INumber<T>
    {
        List<int> histogram = CVProcessing.Histogram(imageIn, bucketCount, out double min, out double max, out double bucketSize);
        int bucket = otsuBucket(imageIn, histogram);
        double otsuThreshold = (double)Convert.ChangeType(min, typeof(double)) + bucketSize * bucket;
        outImage = CVSmaller.Smaller(imageIn, otsuThreshold + 1.0);
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