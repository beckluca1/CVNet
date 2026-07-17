using System.Numerics;
using System.Runtime.InteropServices;

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
        CVImage outImage = CVImage.Create(imageC.Width + 1, imageC.Height + 1, imageC.DataFormat, imageC.ChannelFormats);

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
        CVImage outImage = CVImage.Create(image.Width, image.Height, integralImage.DataFormat, integralImage.ChannelFormats);

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
        CVImage outImage = CVImage.Create(width, height, integralImage.DataFormat, integralImage.ChannelFormats);

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

    public static T MinValue<T>(CVImage imageIn) where T : struct, INumber<T>
    {
        Span<T> buffer = imageIn.BufferAs<T>();

        int length = buffer.Length;

        if (length == 0)
            throw new ArgumentException("Buffer is empty");

        int vectorSize = Vector<T>.Count;
        int i = 0;

        T minValue;

        if (length >= vectorSize)
        {
            Vector<T> minVector = new(buffer);

            i = vectorSize;

            for (; i <= length - vectorSize; i += vectorSize)
            {
                Vector<T> v = new(buffer.Slice(i));

                minVector = Vector.Min(minVector, v);
            }

            minValue = minVector[0];

            for (int j = 1; j < vectorSize; j++)
            {
                if (minVector[j] < minValue)
                    minValue = minVector[j];
            }
        }
        else
        {
            minValue = buffer[0];
            i = 1;
        }

        for (; i < length; i++)
        {
            if (buffer[i] < minValue)
                minValue = buffer[i];
        }

        return minValue;
    }


    public static T MaxValue<T>(CVImage imageIn) where T : struct, INumber<T>
    {
        Span<T> buffer = imageIn.BufferAs<T>();

        int length = buffer.Length;

        if (length == 0)
            throw new ArgumentException("Buffer is empty");

        int vectorSize = Vector<T>.Count;
        int i = 0;

        T maxValue;

        if (length >= vectorSize)
        {
            Vector<T> maxVector = new(buffer);

            i = vectorSize;

            for (; i <= length - vectorSize; i += vectorSize)
            {
                Vector<T> v = new(buffer.Slice(i));

                maxVector = Vector.Max(maxVector, v);
            }

            maxValue = maxVector[0];

            for (int j = 1; j < vectorSize; j++)
            {
                if (maxVector[j] > maxValue)
                    maxValue = maxVector[j];
            }
        }
        else
        {
            maxValue = buffer[0];
            i = 1;
        }

        for (; i < length; i++)
        {
            if (buffer[i] > maxValue)
                maxValue = buffer[i];
        }

        return maxValue;
    }

    private static void normalize<T>(CVImage image, T newMin, T newMax, ref CVImage outImage) where T : struct, INumber<T>
    {
        T min = MinValue<T>(image);
        T max = MaxValue<T>(image);
        T range = max - min;

        T newRange = newMax - newMin;

        outImage = CVSubtract.Subtract(image, min);
        outImage = CVDivide.Divide(outImage, range);
        outImage = CVMultiply.Multiply(outImage, newRange);
        outImage = CVAdd.Add(outImage, newMin);
    }

    public static CVImage Normalize(CVImage image, double min, double max)
    {
        // Normalization requires floating point image
        CVImage imageC = CVConvert.ConvertDataFormatToFloat(image);
        CVImage outImage = CVImage.Create(imageC.Width, imageC.Height, imageC.DataFormat, imageC.ChannelFormats);

        if (imageC.DataFormat == CVDataFormat.CV_U8) normalize<byte>(imageC, (byte)min, (byte)max, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S8) normalize<sbyte>(imageC, (sbyte)min, (sbyte)max, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_U16) normalize<ushort>(imageC, (ushort)min, (ushort)max, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S16) normalize<short>(imageC, (short)min, (short)max, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_U32) normalize<uint>(imageC, (uint)min, (uint)max, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S32) normalize<int>(imageC, (int)min, (int)max, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_U64) normalize<ulong>(imageC, (ulong)min, (ulong)max, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S64) normalize<long>(imageC, (long)min, (long)max, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_F32) normalize<float>(imageC, (float)min, (float)max, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_F64) normalize<double>(imageC, (double)min, (double)max, ref outImage);

        return outImage;
    }

    private static void histogram<T>(CVImage imageIn, int bucketCount, out double min, out double max, out double bucketSize, ref List<int> histogram) where T : struct, INumber<T>
    {
        max = double.CreateChecked(MaxValue<T>(imageIn));
        min = double.CreateChecked(MinValue<T>(imageIn));
        bucketSize = (max - min) / (bucketCount - 1);

        if (bucketSize == 0.0)
            bucketSize = 1.0;

        histogram = Enumerable.Repeat(0, bucketCount).ToList();
        Span<T> buffer = imageIn.BufferAs<T>();

        for (int i = 0; i < imageIn.Width * imageIn.Height * imageIn.Channels; i++)
        {
            int bucket = Convert.ToInt32((double.CreateChecked(buffer[i]) - min) / bucketSize);

            if (bucket < 0)
                bucket = 0;
            else if (bucket >= bucketCount)
                bucket = bucketCount - 1;


            histogram[bucket]++;
        }
    }

    public static List<int> Histogram(CVImage image, int bucketCount, out double min, out double max, out double bucketSize)
    {
        List<int> histogramList = new List<int>();

        min = 0;
        max = 0;
        bucketSize = 0;

        if (image.DataFormat == CVDataFormat.CV_U8) histogram<byte>(image, bucketCount, out min, out max, out bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_S8) histogram<sbyte>(image, bucketCount, out min, out max, out bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_U16) histogram<ushort>(image, bucketCount, out min, out max, out bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_S16) histogram<short>(image, bucketCount, out min, out max, out bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_U32) histogram<uint>(image, bucketCount, out min, out max, out bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_S32) histogram<int>(image, bucketCount, out min, out max, out bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_U64) histogram<ulong>(image, bucketCount, out min, out max, out bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_S64) histogram<long>(image, bucketCount, out min, out max, out bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_F32) histogram<float>(image, bucketCount, out min, out max, out bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_F64) histogram<double>(image, bucketCount, out min, out max, out bucketSize, ref histogramList);

        return histogramList;
    }

    private static void histogram<T>(CVImage imageIn, int bucketCount, double min, double max, double bucketSize, ref List<int> histogram) where T : struct, INumber<T>
    {
        if (bucketSize == 0.0)
            bucketSize = 1.0;

        histogram = Enumerable.Repeat(0, bucketCount).ToList();
        Span<T> buffer = imageIn.BufferAs<T>();

        for (int i = 0; i < imageIn.Width * imageIn.Height * imageIn.Channels; i++)
        {
            int bucket = Convert.ToInt32((double.CreateChecked(buffer[i]) - min) / bucketSize);

            if (bucket < 0)
                bucket = 0;
            else if (bucket >= bucketCount)
                bucket = bucketCount - 1;


            histogram[bucket]++;
        }
    }

    public static List<int> Histogram(CVImage image, int bucketCount, double min, double max, double bucketSize)
    {
        List<int> histogramList = new List<int>();

        min = 0;
        max = 0;
        bucketSize = 0;

        if (image.DataFormat == CVDataFormat.CV_U8) histogram<byte>(image, bucketCount, min, max, bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_S8) histogram<sbyte>(image, bucketCount, min, max, bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_U16) histogram<ushort>(image, bucketCount, min, max, bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_S16) histogram<short>(image, bucketCount, min, max, bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_U32) histogram<uint>(image, bucketCount, min, max, bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_S32) histogram<int>(image, bucketCount, min, max, bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_U64) histogram<ulong>(image, bucketCount, min, max, bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_S64) histogram<long>(image, bucketCount, min, max, bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_F32) histogram<float>(image, bucketCount, min, max, bucketSize, ref histogramList);
        else if (image.DataFormat == CVDataFormat.CV_F64) histogram<double>(image, bucketCount, min, max, bucketSize, ref histogramList);

        return histogramList;
    }

    private static int histogramMap<T, TV>(CVImage imageIn, int bucketCount, TV val) where T : struct, INumber<T> where TV : struct, INumber<TV>
    {
        double max = double.CreateChecked(MaxValue<T>(imageIn));
        double min = double.CreateChecked(MinValue<T>(imageIn));
        double bucketSize = (max - min) / (bucketCount - 1);

        if (bucketSize == 0.0)
            bucketSize = 1.0;

        int bucket = Convert.ToInt32((double.CreateChecked(val) - min) / bucketSize);

        if (bucket < 0)
            bucket = 0;
        else if (bucket >= bucketCount)
            bucket = bucketCount - 1;

        return bucket;
    }

    public static int HistogramMap<TV>(CVImage image, int bucketCount, TV val) where TV : struct, INumber<TV>
    {
        if (image.DataFormat == CVDataFormat.CV_U8) return histogramMap<byte, TV>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_S8) return histogramMap<sbyte, TV>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_U16) return histogramMap<ushort, TV>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_S16) return histogramMap<short, TV>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_U32) return histogramMap<uint, TV>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_S32) return histogramMap<int, TV>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_U64) return histogramMap<ulong, TV>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_S64) return histogramMap<long, TV>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_F32) return histogramMap<float, TV>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_F64) return histogramMap<double, TV>(image, bucketCount, val);

        return 0;
    }

    private static int HistogramMap<TV>(double min, double max, double bucketSize, TV val) where TV : struct, INumber<TV>
    {
        if (bucketSize == 0.0)
            bucketSize = 1.0;

        double valD = double.CreateChecked(val);

        if (valD > max) valD = max;
        if (valD < min) valD = min;

        int bucket = Convert.ToInt32((double.CreateChecked(valD) - min) / bucketSize);

        return bucket;
    }

    public static void getPixels<T, TV>(CVImage image, TV value, ref List<(int, int)> pixelListOut) where T : struct, INumber<T> where TV : struct, INumber<TV>
    {
        T valueC = T.CreateChecked(value);

        Span<T> buffer = image.BufferAs<T>();

        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                if (buffer[x + y * image.Width] == valueC) pixelListOut.Add((x, y));
            }
        }
    }

    public static List<(int, int)> GetPixels<TV>(CVImage image, TV value) where TV : struct, INumber<TV>
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

    public static void getPixels<T, TV>(CVImage mask, CVImage image, TV value, ref List<(int, int, double)> pixelListOut) where T : struct, INumber<T> where TV : struct, INumber<TV>
    {
        T valueC = T.CreateChecked(value);

        Span<T> maskBuffer = mask.BufferAs<T>();
        Span<T> buffer = image.BufferAs<T>();

        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                int index = x + y * image.Width;
                if (maskBuffer[index] == valueC) pixelListOut.Add((x, y, double.CreateChecked(buffer[index])));
            }
        }
    }

    public static List<(int, int, double)> GetPixels<TV>(CVImage mask, CVImage image, TV value) where TV : struct, INumber<TV>
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

    private static void subImage<T>(CVImage image, int startX, int startY, int endX, int endY, ref CVImage imageOut) where T : struct, INumber<T>
    {
        Span<T> src = image.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        for (int c = 0; c < image.Channels; c++)
        {
            for (int y = startY; y < endY; y++)
            {
                for (int x = startX; x < endX; x++)
                {
                    dst[(x - startX) + (y - startY) * imageOut.Width] = src[x + y * image.Width];
                }
            }
        }
    }

    private static CVImage SubImage(CVImage image, int startX, int startY, int endX, int endY)
    {
        CVImage outImage = CVImage.Create(endX - startX, endY - startY, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) subImage<byte>(image, startX, startY, endX, endY, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) subImage<sbyte>(image, startX, startY, endX, endY, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) subImage<ushort>(image, startX, startY, endX, endY, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) subImage<short>(image, startX, startY, endX, endY, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) subImage<uint>(image, startX, startY, endX, endY, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) subImage<int>(image, startX, startY, endX, endY, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) subImage<ulong>(image, startX, startY, endX, endY, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) subImage<long>(image, startX, startY, endX, endY, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) subImage<float>(image, startX, startY, endX, endY, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) subImage<double>(image, startX, startY, endX, endY, ref outImage);

        return outImage;
    }

    private static byte[,] buildLut(
            CVImage image,
            List<int> histogram,
            double clipFactor)
    {
        const int bins = 256;

        int pixels = image.Width * image.Height;

        int clipLimit = Math.Max(1,
            (int)(clipFactor * pixels / bins));

        int excess = 0;

        for (int i = 0; i < bins; i++)
        {
            if (histogram[i] > clipLimit)
            {
                excess += histogram[i] - clipLimit;
                histogram[i] = clipLimit;
            }
        }

        // Redistribute clipped pixels
        int increment = excess / bins;
        int remainder = excess % bins;

        for (int i = 0; i < bins; i++)
            histogram[i] += increment;

        for (int i = 0; i < remainder; i++)
            histogram[i]++;

        // CDF
        int[] cdf = new int[bins];
        cdf[0] = histogram[0];

        for (int i = 1; i < bins; i++)
            cdf[i] = cdf[i - 1] + histogram[i];

        int cdfMin = 0;

        for (int i = 0; i < bins; i++)
        {
            if (cdf[i] != 0)
            {
                cdfMin = cdf[i];
                break;
            }
        }

        byte[,] lut = new byte[1, bins];

        for (int i = 0; i < bins; i++)
        {
            double value =
                (double)(cdf[i] - cdfMin) /
                (pixels - cdfMin);

            value = Math.Max(0, Math.Min(1, value));

            lut[0, i] = (byte)Math.Round(value * 255);
        }

        return lut;
    }

    private static void clahe<T>(
        CVImage image,
        int tileWidth,
        int tileHeight,
        double clipLimit,
        ref CVImage outImage) where T : struct, INumber<T>
    {
        int height = image.Height;
        int width = image.Width;

        int tilesX = (width + tileWidth - 1) / tileWidth;
        int tilesY = (height + tileHeight - 1) / tileHeight;

        byte[][,] luts = new byte[tilesX * tilesY][,];

        // Compute LUT for each tile
        for (int ty = 0; ty < tilesY; ty++)
        {
            for (int tx = 0; tx < tilesX; tx++)
            {
                int x0 = tx * tileWidth;
                int y0 = ty * tileHeight;

                int tw = Math.Min(tileWidth, width - x0);
                int th = Math.Min(tileHeight, height - y0);

                CVImage subImage = SubImage(image, x0, y0, x0 + tw, y0 + th);

                List<int> histogram = Histogram(subImage, 256, 0, 255, 1);

                luts[ty * tilesX + tx] = buildLut(subImage, histogram, clipLimit);
            }
        }

        Span<T> src = image.BufferAs<T>();
        Span<T> dst = outImage.BufferAs<T>();

        // Apply bilinear interpolation
        for (int y = 0; y < height; y++)
        {
            double gy = (double)y / tileHeight - 0.5;
            int ty = (int)Math.Floor(gy);
            double fy = gy - ty;

            ty = Math.Max(0, Math.Min(tilesY - 2, ty));

            for (int x = 0; x < width; x++)
            {
                double gx = (double)x / tileWidth - 0.5;
                int tx = (int)Math.Floor(gx);
                double fx = gx - tx;

                tx = Math.Max(0, Math.Min(tilesX - 2, tx));

                int p = HistogramMap(0, 255, 1, src[x + y * image.Width]);

                byte[,] lut00 = luts[ty * tilesX + tx];
                byte[,] lut10 = luts[ty * tilesX + tx + 1];
                byte[,] lut01 = luts[(ty + 1) * tilesX + tx];
                byte[,] lut11 = luts[(ty + 1) * tilesX + tx + 1];

                double v =
                    (1 - fx) * (1 - fy) * lut00[0, p] +
                    fx * (1 - fy) * lut10[0, p] +
                    (1 - fx) * fy * lut01[0, p] +
                    fx * fy * lut11[0, p];

                dst[x + y * outImage.Width] = T.CreateChecked(v);
            }
        }
    }

    public static CVImage Clahe(
        CVImage image,
        int tileWidth = 64,
        int tileHeight = 64,
        double clipLimit = 2.0)
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) clahe<byte>(image, tileWidth, tileHeight, clipLimit, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) clahe<sbyte>(image, tileWidth, tileHeight, clipLimit, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) clahe<ushort>(image, tileWidth, tileHeight, clipLimit, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) clahe<short>(image, tileWidth, tileHeight, clipLimit, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) clahe<uint>(image, tileWidth, tileHeight, clipLimit, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) clahe<int>(image, tileWidth, tileHeight, clipLimit, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) clahe<ulong>(image, tileWidth, tileHeight, clipLimit, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) clahe<long>(image, tileWidth, tileHeight, clipLimit, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) clahe<float>(image, tileWidth, tileHeight, clipLimit, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) clahe<double>(image, tileWidth, tileHeight, clipLimit, ref outImage);

        return outImage;
    }

    public static int HammingDistance(ulong a, ulong b)
    {
        return BitOperations.PopCount(a ^ b);
    }
}