using System.Numerics;

namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;

public class CVProcessing
{
    public static void IntegralImage<T>(
        CVImage image1,
        ref CVImage imageOut)
        where T : struct, INumber<T>
    {
        Span<T> src = image1.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        int srcWidth = image1.Width;
        int srcHeight = image1.Height;

        int dstWidth = imageOut.Width;

        int srcPlaneSize = srcWidth * srcHeight;
        int dstPlaneSize = imageOut.Width * imageOut.Height;

        for (int c = 0; c < image1.Channels; c++)
        {
            int srcBase = c * srcPlaneSize;
            int dstBase = c * dstPlaneSize;

            for (int y = 0; y < srcHeight; y++)
            {
                int srcRow = srcBase + y * srcWidth;

                // integral image has +1 border
                int dstPrevRow = dstBase + y * dstWidth + 1;
                int dstCurrRow = dstPrevRow + dstWidth;

                T rowSum = T.Zero;

                for (int x = 0; x < srcWidth; x++)
                {
                    rowSum += src[srcRow + x];
                    dst[dstCurrRow + x] = dst[dstPrevRow + x] + rowSum;
                }

            }
        }
    }

    public static CVImage IntegralImage(CVImage image)
    {
        // Integral image can be at max width times height bigger
        CVImage imageC = CVConvert.ConvertDataFormatFactor(image, image.Width * image.Height);
        CVImage outImage = CVImage.Create(imageC.Width + 1, imageC.Height + 1, imageC.ColorFormat, imageC.DataFormat, imageC.ChannelFormat);

        if (imageC.DataFormat == CVDataFormat.CV_U8) IntegralImage<byte>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S8) IntegralImage<sbyte>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_U16) IntegralImage<ushort>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S16) IntegralImage<short>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_U32) IntegralImage<uint>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S32) IntegralImage<int>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_U64) IntegralImage<ulong>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S64) IntegralImage<long>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_F32) IntegralImage<float>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_F64) IntegralImage<double>(imageC, ref outImage);

        return outImage;
    }

    public static void SumWindow<T>(
    CVImage integralImage,
    int size,
    ref CVImage imageOut) where T : struct, INumber<T>
    {
        Span<T> src = integralImage.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        int srcWidth = integralImage.Width;
        int dstWidth = imageOut.Width;
        int dstHeight = imageOut.Height;

        int srcSize = integralImage.Width * integralImage.Height;
        int dstSize = dstWidth * dstHeight;

        int radius = (size - 1) / 2;

        // Precompute x bounds
        int[] left = new int[dstWidth];
        int[] right = new int[dstWidth];

        for (int x = 0; x < dstWidth; x++)
        {
            left[x] = Math.Max(0, x - radius);
            right[x] = Math.Min(dstWidth - 1, x + radius) + 1;
        }

        int[] top = new int[dstHeight];
        int[] bottom = new int[dstHeight];

        for (int y = 0; y < dstHeight; y++)
        {
            top[y] = Math.Max(0, y - radius);
            bottom[y] = Math.Min(dstHeight - 1, y + radius) + 1;
        }

        for (int c = 0; c < integralImage.Channels; c++)
        {
            int srcChannelOffset = c * srcSize;
            int dstChannelOffset = c * dstSize;

            for (int y = 0; y < dstHeight; y++)
            {
                int rowAB = srcChannelOffset + top[y] * srcWidth;
                int rowCD = srcChannelOffset + bottom[y] * srcWidth;

                int dstRow = dstChannelOffset + y * dstWidth;

                for (int x = 0; x < dstWidth; x++)
                {
                    int x0 = left[x];
                    int x1 = right[x];

                    dst[dstRow + x] =
                        src[rowCD + x1]
                      - src[rowAB + x1]
                      - src[rowCD + x0]
                      + src[rowAB + x0];
                }
            }
        }
    }

    public static CVImage SumWindow(CVImage image, int size)
    {
        CVImage integralImage = IntegralImage(image);
        CVImage outImage = CVImage.Create(image.Width, image.Height, integralImage.ColorFormat, integralImage.DataFormat, integralImage.ChannelFormat);

        if (integralImage.DataFormat == CVDataFormat.CV_U8) SumWindow<byte>(integralImage, size, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_S8) SumWindow<sbyte>(integralImage, size, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_U16) SumWindow<ushort>(integralImage, size, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_S16) SumWindow<short>(integralImage, size, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_U32) SumWindow<uint>(integralImage, size, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_S32) SumWindow<int>(integralImage, size, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_U64) SumWindow<ulong>(integralImage, size, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_S64) SumWindow<long>(integralImage, size, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_F32) SumWindow<float>(integralImage, size, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_F64) SumWindow<double>(integralImage, size, ref outImage);

        return outImage;
    }

    public static void Int32AverageWindowResample(
    CVImage integralImage,
    ref CVImage imageOut)
    {
        Span<int> src = integralImage.BufferAs<int>();
        Span<int> dst = imageOut.BufferAs<int>();

        int srcWidth = integralImage.Width - 1;
        int srcHeight = integralImage.Height - 1;

        int iiWidth = integralImage.Width;
        int iiHeight = integralImage.Height;

        int dstWidth = imageOut.Width;
        int dstHeight = imageOut.Height;

        int srcSize = srcWidth * srcHeight;
        int dstSize = dstWidth * dstHeight;

        int[] x0 = new int[dstWidth];
        int[] x1 = new int[dstWidth];

        for (int x = 0; x < dstWidth; x++)
        {
            x0[x] = (int)(x * (long)srcWidth / dstWidth);
            x1[x] = (int)((x + 1) * (long)srcWidth / dstWidth);

            x0[x] = Math.Clamp(x0[x], 0, srcWidth);
            x1[x] = Math.Clamp(x1[x], 0, srcWidth);
        }

        int[] y0 = new int[dstHeight];
        int[] y1 = new int[dstHeight];

        for (int y = 0; y < dstHeight; y++)
        {
            y0[y] = (int)(y * (long)srcHeight / dstHeight);
            y1[y] = (int)((y + 1) * (long)srcHeight / dstHeight);

            y0[y] = Math.Clamp(y0[y], 0, srcHeight);
            y1[y] = Math.Clamp(y1[y], 0, srcHeight);
        }

        for (int c = 0; c < integralImage.Channels; c++)
        {
            int srcChannelOffset = c * (iiWidth * iiHeight);
            int dstChannelOffset = c * dstSize;

            for (int y = 0; y < dstHeight; y++)
            {
                int ya0 = y0[y];
                int ya1 = y1[y];

                int rowA = srcChannelOffset + ya0 * iiWidth;
                int rowB = srcChannelOffset + ya1 * iiWidth;

                int dstRow = dstChannelOffset + y * dstWidth;

                for (int x = 0; x < dstWidth; x++)
                {
                    int xa0 = x0[x];
                    int xa1 = x1[x];

                    int sum =
                        src[rowB + xa1]
                      - src[rowA + xa1]
                      - src[rowB + xa0]
                      + src[rowA + xa0];

                    int width = xa1 - xa0;
                    int height = ya1 - ya0;
                    int area = width * height;

                    dst[dstRow + x] = sum / area;
                }
            }
        }
    }

    public static void Float32AverageWindowResample(
        CVImage integralImage,
        ref CVImage imageOut)
    {
        Span<float> src = integralImage.BufferAs<float>();
        Span<float> dst = imageOut.BufferAs<float>();

        int srcWidth = integralImage.Width - 1;
        int srcHeight = integralImage.Height - 1;

        int iiWidth = integralImage.Width;
        int iiHeight = integralImage.Height;

        int dstWidth = imageOut.Width;
        int dstHeight = imageOut.Height;

        int srcSize = srcWidth * srcHeight;
        int dstSize = dstWidth * dstHeight;

        int[] x0 = new int[dstWidth];
        int[] x1 = new int[dstWidth];

        for (int x = 0; x < dstWidth; x++)
        {
            x0[x] = (int)(x * (long)srcWidth / dstWidth);
            x1[x] = (int)((x + 1) * (long)srcWidth / dstWidth);

            x0[x] = Math.Clamp(x0[x], 0, srcWidth);
            x1[x] = Math.Clamp(x1[x], 0, srcWidth);
        }

        int[] y0 = new int[dstHeight];
        int[] y1 = new int[dstHeight];

        for (int y = 0; y < dstHeight; y++)
        {
            y0[y] = (int)(y * (long)srcHeight / dstHeight);
            y1[y] = (int)((y + 1) * (long)srcHeight / dstHeight);

            y0[y] = Math.Clamp(y0[y], 0, srcHeight);
            y1[y] = Math.Clamp(y1[y], 0, srcHeight);
        }

        for (int c = 0; c < integralImage.Channels; c++)
        {
            int srcChannelOffset = c * (iiWidth * iiHeight);
            int dstChannelOffset = c * dstSize;

            for (int y = 0; y < dstHeight; y++)
            {
                int ya0 = y0[y];
                int ya1 = y1[y];

                int rowA = srcChannelOffset + ya0 * iiWidth;
                int rowB = srcChannelOffset + ya1 * iiWidth;

                int dstRow = dstChannelOffset + y * dstWidth;

                for (int x = 0; x < dstWidth; x++)
                {
                    int xa0 = x0[x];
                    int xa1 = x1[x];

                    float sum =
                        src[rowB + xa1]
                      - src[rowA + xa1]
                      - src[rowB + xa0]
                      + src[rowA + xa0];

                    int width = xa1 - xa0;
                    int height = ya1 - ya0;
                    int area = width * height;

                    dst[dstRow + x] = sum / area;
                }
            }
        }
    }

    public static void SumWindowResample<T>(
            CVImage integralImage,
            ref CVImage imageOut) where T : struct, INumber<T>
    {
        Span<T> src = integralImage.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        int srcWidth = integralImage.Width - 1;
        int srcHeight = integralImage.Height - 1;

        int iiWidth = integralImage.Width;
        int iiHeight = integralImage.Height;

        int dstWidth = imageOut.Width;
        int dstHeight = imageOut.Height;

        int srcSize = srcWidth * srcHeight;
        int dstSize = dstWidth * dstHeight;

        int[] x0 = new int[dstWidth];
        int[] x1 = new int[dstWidth];

        //T area = (T)Convert.ChangeType(((double)srcWidth / dstWidth) * ((double)srcHeight / dstHeight), typeof(T));

        for (int x = 0; x < dstWidth; x++)
        {
            x0[x] = (int)(x * (long)srcWidth / dstWidth);
            x1[x] = (int)((x + 1) * (long)srcWidth / dstWidth);

            x0[x] = Math.Clamp(x0[x], 0, srcWidth);
            x1[x] = Math.Clamp(x1[x], 0, srcWidth);
        }

        int[] y0 = new int[dstHeight];
        int[] y1 = new int[dstHeight];

        for (int y = 0; y < dstHeight; y++)
        {
            y0[y] = (int)(y * (long)srcHeight / dstHeight);
            y1[y] = (int)((y + 1) * (long)srcHeight / dstHeight);

            y0[y] = Math.Clamp(y0[y], 0, srcHeight);
            y1[y] = Math.Clamp(y1[y], 0, srcHeight);
        }

        for (int c = 0; c < integralImage.Channels; c++)
        {
            int srcChannelOffset = c * (iiWidth * iiHeight);
            int dstChannelOffset = c * dstSize;

            for (int y = 0; y < dstHeight; y++)
            {
                int ya0 = y0[y];
                int ya1 = y1[y];

                int rowA = srcChannelOffset + ya0 * iiWidth;
                int rowB = srcChannelOffset + ya1 * iiWidth;

                int dstRow = dstChannelOffset + y * dstWidth;

                for (int x = 0; x < dstWidth; x++)
                {
                    int xa0 = x0[x];
                    int xa1 = x1[x];

                    int width = xa1 - xa0;
                    int height = ya1 - ya0;
                    int area = width * height;

                    T sum =
                        src[rowB + xa1]
                      - src[rowA + xa1]
                      - src[rowB + xa0]
                      + src[rowA + xa0];

                    dst[dstRow + x] = sum;
                }
            }
        }
    }

    public static CVImage SumWindowResample(CVImage image, int width, int height)
    {
        // Average Window requires floating point number
        CVImage integralImage = IntegralImage(image);
        CVImage outImage = CVImage.Create(width, height, integralImage.ColorFormat, integralImage.DataFormat, integralImage.ChannelFormat);

        if (integralImage.DataFormat == CVDataFormat.CV_U8) SumWindowResample<byte>(integralImage, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_S8) SumWindowResample<sbyte>(integralImage, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_U16) SumWindowResample<ushort>(integralImage, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_S16) SumWindowResample<short>(integralImage, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_U32) SumWindowResample<uint>(integralImage, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_S32) SumWindowResample<int>(integralImage, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_U64) SumWindowResample<ulong>(integralImage, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_S64) SumWindowResample<long>(integralImage, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_F32) SumWindowResample<float>(integralImage, ref outImage);
        else if (integralImage.DataFormat == CVDataFormat.CV_F64) SumWindowResample<double>(integralImage, ref outImage);

        //if (integralImage.DataFormat == CVDataFormat.CV_S32) Int32AverageWindowResample(integralImage, ref outImage);
        //else if (integralImage.DataFormat == CVDataFormat.CV_F32) Float32AverageWindowResample(integralImage, ref outImage);

        return outImage;
    }

    public static T MaxValue<T>(CVImage imageIn) where T : struct, INumber<T>
    {
        Span<T> buffer = imageIn.BufferAs<T>();
        T maxValue = buffer[0];
        for (int i = 1; i < imageIn.Width * imageIn.Height * imageIn.Channels; i++)
            if (buffer[i] > maxValue) maxValue = buffer[i];

        return maxValue;
    }

    public static T MinValue<T>(CVImage imageIn) where T : struct, INumber<T>
    {
        Span<T> buffer = imageIn.BufferAs<T>();
        T minValue = buffer[0];
        for (int i = 1; i < imageIn.Width * imageIn.Height * imageIn.Channels; i++)
            if (buffer[i] < minValue) minValue = buffer[i];

        return minValue;
    }

    private static void normalize<T>(CVImage image, ref CVImage outImage) where T : struct, INumber<T>
    {
        T min = MinValue<T>(image);
        T max = MinValue<T>(image);
        T range = max - min;

        outImage = CVSubtract.Subtract(image, min);
        outImage = CVDivide.Divide(outImage, range);
    }

    public static CVImage Normalize(CVImage image)
    {
        // Normalization requires floating point image
        CVImage imageC = CVConvert.ConvertDataFormatToFloat(image);
        CVImage outImage = CVImage.Create(imageC.Width, imageC.Height, imageC.ColorFormat, imageC.DataFormat, imageC.ChannelFormat);

        if (imageC.DataFormat == CVDataFormat.CV_U8) normalize<byte>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S8) normalize<sbyte>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_U16) normalize<ushort>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S16) normalize<short>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_U32) normalize<uint>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S32) normalize<int>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_U64) normalize<ulong>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S64) normalize<long>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_F32) normalize<float>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_F64) normalize<double>(imageC, ref outImage);

        return outImage;
    }

    public static List<int> Histogram<T>(CVImage imageIn, int bucketCount, out T min, out T max, out double bucketSize) where T : struct, INumber<T>
    {
        max = MaxValue<T>(imageIn);
        min = MinValue<T>(imageIn);
        T values = max - min;

        bucketSize = (double)Convert.ChangeType(values, typeof(double)) / (bucketCount - 1);

        if (bucketSize == 0.0)
            bucketSize = 1.0;

        List<int> histogram = Enumerable.Repeat(0, bucketCount).ToList();
        Span<T> buffer = imageIn.BufferAs<T>();

        for (int i = 0; i < imageIn.Width * imageIn.Height * imageIn.Channels; i++)
        {
            int bucket = Convert.ToInt32((double)Convert.ChangeType(buffer[i] - min, typeof(double)) / bucketSize);

            if (bucket < 0)
                bucket = 0;
            else if (bucket >= bucketCount)
                bucket = bucketCount - 1;


            histogram[bucket]++;
        }

        return histogram;
    }


    public static void getPixels<T, TV>(CVImage image, TV value, ref List<(int, int)> pixelListOut) where T : struct, INumber<T> where TV : struct
    {
        T valueC = (T)Convert.ChangeType(value, typeof(T));

        Span<T> buffer = image.BufferAs<T>();

        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                if (buffer[x + y * image.Width] == valueC) pixelListOut.Add((x, y));
            }
        }
    }

    public static List<(int, int)> GetPixels<TV>(CVImage image, TV value) where TV : struct
    {
        List<(int, int)> pixelListOut = new List<(int, int)>();

        if (image.DataFormat == CVDataFormat.CV_U8) getPixels<byte, TV>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) getPixels<sbyte, TV>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) getPixels<ushort, TV>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) getPixels<short, TV>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) getPixels<uint, TV>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) getPixels<int, TV>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) getPixels<ulong, TV>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) getPixels<long, TV>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) getPixels<float, TV>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) getPixels<double, TV>(image, value, ref pixelListOut);

        return pixelListOut;
    }

    public static void getPixels<T, TV>(CVImage mask, CVImage image, TV value, ref List<(int, int, double)> pixelListOut) where T : struct, INumber<T> where TV : struct
    {
        T valueC = (T)Convert.ChangeType(value, typeof(T));

        Span<T> maskBuffer = mask.BufferAs<T>();
        Span<T> buffer = image.BufferAs<T>();

        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                int index = x + y * image.Width;
                if (maskBuffer[index] == valueC) pixelListOut.Add((x, y, (double)Convert.ChangeType(buffer[index], typeof(double))));
            }
        }
    }

    public static List<(int, int, double)> GetPixels<TV>(CVImage mask, CVImage image, TV value) where TV : struct
    {
        List<(int, int, double)> pixelListOut = new List<(int, int, double)>();

        if (image.DataFormat == CVDataFormat.CV_U8) getPixels<byte, TV>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) getPixels<sbyte, TV>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) getPixels<ushort, TV>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) getPixels<short, TV>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) getPixels<uint, TV>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) getPixels<int, TV>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) getPixels<ulong, TV>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) getPixels<long, TV>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) getPixels<float, TV>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) getPixels<double, TV>(mask, image, value, ref pixelListOut);

        return pixelListOut;
    }
}