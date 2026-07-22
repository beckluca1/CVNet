using System.Numerics;
using System.Runtime.InteropServices;

namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;

public class CVProcessing
{
    private static T minValue<T>(CVImage imageIn) where T : struct, INumber<T>
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

    public static double MinValue(CVImage image)
    {
        double value = 0.0;

        if (image.DataFormat == CVDataFormat.CV_U8) value = minValue<byte>(image);
        else if (image.DataFormat == CVDataFormat.CV_S8) value = minValue<sbyte>(image);
        else if (image.DataFormat == CVDataFormat.CV_U16) value = minValue<ushort>(image);
        else if (image.DataFormat == CVDataFormat.CV_S16) value = minValue<short>(image);
        else if (image.DataFormat == CVDataFormat.CV_U32) value = minValue<uint>(image);
        else if (image.DataFormat == CVDataFormat.CV_S32) value = minValue<int>(image);
        else if (image.DataFormat == CVDataFormat.CV_U64) value = minValue<ulong>(image);
        else if (image.DataFormat == CVDataFormat.CV_S64) value = minValue<long>(image);
        else if (image.DataFormat == CVDataFormat.CV_F32) value = minValue<float>(image);
        else if (image.DataFormat == CVDataFormat.CV_F64) value = minValue<double>(image);

        return value;
    }

    public static T maxValue<T>(CVImage imageIn) where T : struct, INumber<T>
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

    public static double MaxValue(CVImage image)
    {
        double value = 0.0;

        if (image.DataFormat == CVDataFormat.CV_U8) value = maxValue<byte>(image);
        else if (image.DataFormat == CVDataFormat.CV_S8) value = maxValue<sbyte>(image);
        else if (image.DataFormat == CVDataFormat.CV_U16) value = maxValue<ushort>(image);
        else if (image.DataFormat == CVDataFormat.CV_S16) value = maxValue<short>(image);
        else if (image.DataFormat == CVDataFormat.CV_U32) value = maxValue<uint>(image);
        else if (image.DataFormat == CVDataFormat.CV_S32) value = maxValue<int>(image);
        else if (image.DataFormat == CVDataFormat.CV_U64) value = maxValue<ulong>(image);
        else if (image.DataFormat == CVDataFormat.CV_S64) value = maxValue<long>(image);
        else if (image.DataFormat == CVDataFormat.CV_F32) value = maxValue<float>(image);
        else if (image.DataFormat == CVDataFormat.CV_F64) value = maxValue<double>(image);

        return value;
    }

    private static void normalize(CVImage image, double newMin, double newMax, ref CVImage outImage)
    {
        double min = MinValue(image);
        double max = MaxValue(image);
        double range = max - min;

        double newRange = newMax - newMin;

        outImage = CVSubtract.Subtract(image, min);
        if (range > 10e-12)
        {
            outImage = CVDivide.Divide(outImage, range);
            outImage = CVMultiply.Multiply(outImage, newRange);
        }
        outImage = CVAdd.Add(outImage, newMin);
    }

    public static CVImage Normalize(CVImage image, double min, double max)
    {
        CVImage doubleImage = CVConvert.ConvertDataFormat(image, CVDataFormat.CV_F64);
        CVImage outImage = CVImage.Create(image.Width, image.Height, CVDataFormat.CV_F64, image.ChannelFormats);

        normalize(doubleImage, min, max, ref outImage);

        return outImage;
    }

    private static void histogram<T>(CVImage imageIn, int bucketCount, out double min, out double max, out double bucketSize, ref CVImage histogramImage) where T : struct, INumber<T>
    {
        max = MaxValue(imageIn);
        min = MinValue(imageIn);
        bucketSize = (max - min) / (bucketCount - 1);

        if (bucketSize == 0.0)
            bucketSize = 1.0;

        Span<uint> hBuffer = histogramImage.BufferAs<uint>();
        Span<T> buffer = imageIn.BufferAs<T>();

        int srcPlaneSize = imageIn.Width * imageIn.Height;
        int dstPlaneSize = histogramImage.Width * histogramImage.Height;

        for (int c = 0; c < imageIn.Channels; c++)
        {
            int srcChannelOffset = c * srcPlaneSize;
            int dstChannelOffset = c * dstPlaneSize;
            for (int i = 0; i < imageIn.Width * imageIn.Height; i++)
            {
                int bucket = Convert.ToInt32((double.CreateChecked(buffer[srcChannelOffset + i]) - min) / bucketSize);

                if (bucket < 0)
                    bucket = 0;
                else if (bucket >= bucketCount)
                    bucket = bucketCount - 1;


                hBuffer[dstChannelOffset + bucket]++;
            }
        }
    }

    public static CVImage Histogram(CVImage image, int bucketCount, out double min, out double max, out double bucketSize)
    {
        CVImage histogramImage = CVImage.Create(bucketCount, 1, CVDataFormat.CV_U32, image.ChannelFormats);

        min = 0;
        max = 0;
        bucketSize = 0;

        if (image.DataFormat == CVDataFormat.CV_U8) histogram<byte>(image, bucketCount, out min, out max, out bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) histogram<sbyte>(image, bucketCount, out min, out max, out bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) histogram<ushort>(image, bucketCount, out min, out max, out bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) histogram<short>(image, bucketCount, out min, out max, out bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) histogram<uint>(image, bucketCount, out min, out max, out bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) histogram<int>(image, bucketCount, out min, out max, out bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) histogram<ulong>(image, bucketCount, out min, out max, out bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) histogram<long>(image, bucketCount, out min, out max, out bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) histogram<float>(image, bucketCount, out min, out max, out bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) histogram<double>(image, bucketCount, out min, out max, out bucketSize, ref histogramImage);

        return histogramImage;
    }

    private static void histogram<T>(CVImage imageIn, int bucketCount, double min, double max, double bucketSize, ref CVImage histogramImage) where T : struct, INumber<T>
    {
        if (bucketSize == 0.0)
            bucketSize = 1.0;

        Span<uint> hBuffer = histogramImage.BufferAs<uint>();
        Span<T> buffer = imageIn.BufferAs<T>();

        int srcPlaneSize = imageIn.Width * imageIn.Height;
        int dstPlaneSize = histogramImage.Width * histogramImage.Height;

        for (int c = 0; c < imageIn.Channels; c++)
        {
            int srcChannelOffset = c * srcPlaneSize;
            int dstChannelOffset = c * dstPlaneSize;
            for (int i = 0; i < imageIn.Width * imageIn.Height; i++)
            {
                int bucket = Convert.ToInt32((double.CreateChecked(buffer[srcChannelOffset + i]) - min) / bucketSize);

                if (bucket < 0)
                    bucket = 0;
                else if (bucket >= bucketCount)
                    bucket = bucketCount - 1;


                hBuffer[dstChannelOffset + bucket]++;
            }
        }
    }

    public static CVImage Histogram(CVImage image, int bucketCount, double min, double max, double bucketSize)
    {
        CVImage histogramImage = CVImage.Create(bucketCount, 1, CVDataFormat.CV_U32, image.ChannelFormats);

        min = 0;
        max = 0;
        bucketSize = 0;

        if (image.DataFormat == CVDataFormat.CV_U8) histogram<byte>(image, bucketCount, min, max, bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) histogram<sbyte>(image, bucketCount, min, max, bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) histogram<ushort>(image, bucketCount, min, max, bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) histogram<short>(image, bucketCount, min, max, bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) histogram<uint>(image, bucketCount, min, max, bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) histogram<int>(image, bucketCount, min, max, bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) histogram<ulong>(image, bucketCount, min, max, bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) histogram<long>(image, bucketCount, min, max, bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) histogram<float>(image, bucketCount, min, max, bucketSize, ref histogramImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) histogram<double>(image, bucketCount, min, max, bucketSize, ref histogramImage);

        return histogramImage;
    }

    private static int histogramMap<T, TV>(CVImage imageIn, int bucketCount, TV val) where T : struct, INumber<T> where TV : struct, INumber<TV>
    {
        double max = MaxValue(imageIn);
        double min = MinValue(imageIn);
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

    public static int HistogramMap<T>(CVImage image, int bucketCount, T val) where T : struct, INumber<T>
    {
        if (image.DataFormat == CVDataFormat.CV_U8) return histogramMap<byte, T>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_S8) return histogramMap<sbyte, T>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_U16) return histogramMap<ushort, T>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_S16) return histogramMap<short, T>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_U32) return histogramMap<uint, T>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_S32) return histogramMap<int, T>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_U64) return histogramMap<ulong, T>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_S64) return histogramMap<long, T>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_F32) return histogramMap<float, T>(image, bucketCount, val);
        else if (image.DataFormat == CVDataFormat.CV_F64) return histogramMap<double, T>(image, bucketCount, val);

        return 0;
    }

    private static int HistogramMap<T>(double min, double max, double bucketSize, T val) where T : struct, INumber<T>
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

    public static List<(int, int)> GetPixels<T>(CVImage image, T value) where T : struct, INumber<T>
    {
        List<(int, int)> pixelListOut = new List<(int, int)>();

        if (image.DataFormat == CVDataFormat.CV_U8) getPixels<byte, T>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) getPixels<sbyte, T>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) getPixels<ushort, T>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) getPixels<short, T>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) getPixels<uint, T>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) getPixels<int, T>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) getPixels<ulong, T>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) getPixels<long, T>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) getPixels<float, T>(image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) getPixels<double, T>(image, value, ref pixelListOut);

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

    public static List<(int, int, double)> GetPixels<T>(CVImage mask, CVImage image, T value) where T : struct, INumber<T>
    {
        List<(int, int, double)> pixelListOut = new List<(int, int, double)>();

        if (image.DataFormat == CVDataFormat.CV_U8) getPixels<byte, T>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) getPixels<sbyte, T>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) getPixels<ushort, T>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) getPixels<short, T>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) getPixels<uint, T>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) getPixels<int, T>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) getPixels<ulong, T>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) getPixels<long, T>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) getPixels<float, T>(mask, image, value, ref pixelListOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) getPixels<double, T>(mask, image, value, ref pixelListOut);

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
            CVImage histogram,
            double clipFactor)
    {
        Span<uint> hBuffer = histogram.BufferAs<uint>();

        const int bins = 256;

        int pixels = image.Width * image.Height;

        int clipLimit = Math.Max(1,
            (int)(clipFactor * pixels / bins));

        int excess = 0;

        for (int i = 0; i < bins; i++)
        {
            if (hBuffer[i] > clipLimit)
            {
                excess += (int)(hBuffer[i] - clipLimit);
                hBuffer[i] = (uint)clipLimit;
            }
        }

        // Redistribute clipped pixels
        int increment = excess / bins;
        int remainder = excess % bins;

        for (int i = 0; i < bins; i++)
            hBuffer[i] += (uint)increment;

        for (int i = 0; i < remainder; i++)
            hBuffer[i]++;

        // CDF
        int[] cdf = new int[bins];
        cdf[0] = (int)hBuffer[0];

        for (int i = 1; i < bins; i++)
            cdf[i] = (int)(cdf[i - 1] + hBuffer[i]);

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

                CVImage histogram = Histogram(subImage, 256, 0, 255, 1);

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