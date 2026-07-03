using System.Numerics;

namespace CVNet;

public class CVProcessing
{
    public static void Int32IntegralImage<T>(
        CVImage image1,
        ref CVImage imageOut)
        where T : struct, INumber<T>
    {
        Span<T> src = image1.BufferAs<T>();
        Span<int> dst = imageOut.BufferAs<int>();

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

                int rowSum = 0;

                for (int x = 0; x < srcWidth; x++)
                {
                    rowSum += (int)Convert.ChangeType(src[srcRow + x], typeof(int));
                    dst[dstCurrRow + x] = dst[dstPrevRow + x] + rowSum;
                }
            }
        }
    }

    public static void Float32IntegralImage<T>(
            CVImage image1,
            ref CVImage imageOut)
            where T : struct, INumber<T>
    {
        Span<T> src = image1.BufferAs<T>();
        Span<float> dst = imageOut.BufferAs<float>();

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

                float rowSum = 0.0f;

                for (int x = 0; x < srcWidth; x++)
                {
                    rowSum += (float)Convert.ChangeType(src[srcRow + x], typeof(float));
                    dst[dstCurrRow + x] = dst[dstPrevRow + x] + rowSum;
                }
            }
        }
    }

    public static void IntegralImage<T>(CVImage image1, CVDataFormat dataFormat, ref CVImage outImage) where T : struct, INumber<T>
    {
        if (dataFormat == CVDataFormat.CV_S32) Int32IntegralImage<T>(image1, ref outImage);
        else if (dataFormat == CVDataFormat.CV_F32) Float32IntegralImage<T>(image1, ref outImage);
    }

    public static CVImage IntegralImage(CVImage image1, CVDataFormat dataFormat)
    {
        // Integral image always has type Float32
        CVImage outImage = CVImage.Create(image1.Width + 1, image1.Height + 1, image1.ColorFormat, dataFormat, image1.ChannelFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) IntegralImage<byte>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) IntegralImage<sbyte>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) IntegralImage<ushort>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) IntegralImage<short>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) IntegralImage<uint>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) IntegralImage<int>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) IntegralImage<ulong>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) IntegralImage<long>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) IntegralImage<float>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) IntegralImage<double>(image1, dataFormat, ref outImage);

        return outImage;
    }

    public static void Int32SumWindow<T>(
    CVImage image1,
    int size,
    ref CVImage imageOut)
    where T : struct, INumber<T>
    {
        CVImage integralImage = IntegralImage(image1, CVDataFormat.CV_S32);

        Span<T> src = integralImage.BufferAs<T>();
        Span<int> dst = imageOut.BufferAs<int>();

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

        for (int c = 0; c < image1.Channels; c++)
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

                    dst[dstRow + x] = (int)Convert.ChangeType(
                        src[rowCD + x1]
                      - src[rowAB + x1]
                      - src[rowCD + x0]
                      + src[rowAB + x0], typeof(int));
                }
            }
        }
    }

    public static void Float32SumWindow<T>(
        CVImage image1,
        int size,
        ref CVImage imageOut)
        where T : struct, INumber<T>
    {
        CVImage integralImage = IntegralImage(image1, CVDataFormat.CV_S32);

        Span<T> src = integralImage.BufferAs<T>();
        Span<float> dst = imageOut.BufferAs<float>();

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

        for (int c = 0; c < image1.Channels; c++)
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

                    dst[dstRow + x] = (float)Convert.ChangeType(
                        src[rowCD + x1]
                      - src[rowAB + x1]
                      - src[rowCD + x0]
                      + src[rowAB + x0], typeof(float));
                }
            }
        }
    }

    public static void SumWindow<T>(CVImage image1, int size, CVDataFormat dataFormat, ref CVImage outImage) where T : struct, INumber<T>
    {
        if (dataFormat == CVDataFormat.CV_S32) Int32SumWindow<T>(image1, size, ref outImage);
        else if (dataFormat == CVDataFormat.CV_F32) Float32SumWindow<T>(image1, size, ref outImage);
    }

    public static CVImage SumWindow(CVImage image1, int size, CVDataFormat dataFormat)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, dataFormat, image1.ChannelFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) SumWindow<byte>(image1, size, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) SumWindow<sbyte>(image1, size, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) SumWindow<ushort>(image1, size, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) SumWindow<short>(image1, size, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) SumWindow<uint>(image1, size, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) SumWindow<int>(image1, size, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) SumWindow<ulong>(image1, size, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) SumWindow<long>(image1, size, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) SumWindow<float>(image1, size, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) SumWindow<double>(image1, size, dataFormat, ref outImage);

        return outImage;
    }

    public static void Int32AverageWindowResample<T>(
    CVImage image1,
    ref CVImage imageOut)
    where T : struct, INumber<T>
    {
        CVImage integralImage = IntegralImage(image1, CVDataFormat.CV_S32);

        Span<T> src = integralImage.BufferAs<T>();
        Span<int> dst = imageOut.BufferAs<int>();

        int srcWidth = image1.Width;
        int srcHeight = image1.Height;

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

        for (int c = 0; c < image1.Channels; c++)
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

                    int sum = (int)Convert.ChangeType(
                        src[rowB + xa1]
                      - src[rowA + xa1]
                      - src[rowB + xa0]
                      + src[rowA + xa0], typeof(int));

                    int width = xa1 - xa0;
                    int height = ya1 - ya0;
                    int area = width * height;

                    dst[dstRow + x] = sum / area;
                }
            }
        }
    }

    public static void Float32AverageWindowResample<T>(
        CVImage image1,
        ref CVImage imageOut)
        where T : struct, INumber<T>
    {
        CVImage integralImage = IntegralImage(image1, CVDataFormat.CV_S32);

        Span<T> src = integralImage.BufferAs<T>();
        Span<float> dst = imageOut.BufferAs<float>();

        int srcWidth = image1.Width;
        int srcHeight = image1.Height;

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

        for (int c = 0; c < image1.Channels; c++)
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

                    float sum = (float)Convert.ChangeType(
                        src[rowB + xa1]
                      - src[rowA + xa1]
                      - src[rowB + xa0]
                      + src[rowA + xa0], typeof(float));

                    int width = xa1 - xa0;
                    int height = ya1 - ya0;
                    int area = width * height;

                    dst[dstRow + x] = sum / area;
                }
            }
        }
    }

    public static void AverageWindowResample<T>(CVImage image1, CVDataFormat dataFormat, ref CVImage outImage) where T : struct, INumber<T>
    {
        if (dataFormat == CVDataFormat.CV_S32) Int32AverageWindowResample<T>(image1, ref outImage);
        else if (dataFormat == CVDataFormat.CV_F32) Float32AverageWindowResample<T>(image1, ref outImage);
    }

    public static CVImage AverageWindowResample(CVImage image1, int width, int height, CVDataFormat dataFormat)
    {
        CVImage outImage = CVImage.Create(width, height, image1.ColorFormat, dataFormat, image1.ChannelFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) AverageWindowResample<byte>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) AverageWindowResample<sbyte>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) AverageWindowResample<ushort>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) AverageWindowResample<short>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) AverageWindowResample<uint>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) AverageWindowResample<int>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) AverageWindowResample<ulong>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) AverageWindowResample<long>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) AverageWindowResample<float>(image1, dataFormat, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) AverageWindowResample<double>(image1, dataFormat, ref outImage);

        return outImage;
    }

    public static void ConvolutionX<T>(
        CVImage image1,
        CVImage image2,
        ref CVImage imageOut)
        where T : struct,
            INumber<T>
    {
        int kW = (image2.Width - 1) / 2;

        Span<T> src = image1.BufferAs<T>();
        Span<T> kernel = image2.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        int width = image1.Width;
        int height = image1.Height;

        int simdWidth = Vector<T>.Count;

        // assume kernel is 1D horizontal
        int kWidth = image2.Width;

        for (int y = 0; y < height; y++)
        {
            int row = y * width;

            int x = kW;

            // SIMD main loop
            for (; x <= width - kW - simdWidth; x += simdWidth)
            {
                Vector<T> acc = Vector<T>.Zero;

                for (int dx = -kW; dx <= kW; dx++)
                {
                    T k = kernel[dx + kW];

                    int srcIndex = row + x + dx;

                    Vector<T> vec = Vector.LoadUnsafe(ref src[srcIndex]);

                    acc += vec * k;
                }

                acc.CopyTo(dst.Slice(row + x));
            }

            // scalar tail
            for (; x < width - kW; x++)
            {
                T sum = T.AdditiveIdentity;

                for (int dx = -kW; dx <= kW; dx++)
                {
                    sum += src[row + x + dx] * kernel[dx + kW];
                }

                dst[row + x] = sum;
            }
        }
    }

    public static void ConvolutionY<T>(
        CVImage image1,
        CVImage image2,
        ref CVImage imageOut)
        where T : struct,
            INumber<T>
    {
        int kH = (image2.Height - 1) / 2;
        int width = image1.Width;
        int height = image1.Height;

        Span<T> src = image1.BufferAs<T>();
        Span<T> kernel = image2.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        int simdWidth = Vector<T>.Count;

        for (int y = kH; y < height - kH; y++)
        {
            int rowOut = y * width;

            int x = 0;

            // SIMD over X
            for (; x <= width - simdWidth; x += simdWidth)
            {
                Vector<T> acc = Vector<T>.Zero;

                for (int dy = -kH; dy <= kH; dy++)
                {
                    int rowIn = (y + dy) * width;
                    T k = kernel[dy + kH];

                    var v = Vector.LoadUnsafe(ref src[rowIn + x]);

                    acc += v * k;
                }

                acc.CopyTo(dst.Slice(rowOut + x));
            }

            // scalar tail
            for (; x < width; x++)
            {
                T sum = T.AdditiveIdentity;

                for (int dy = -kH; dy <= kH; dy++)
                {
                    int rowIn = (y + dy) * width;
                    sum += src[rowIn + x] * kernel[dy + kH];
                }

                dst[rowOut + x] = sum;
            }
        }
    }

    // Optimized
    public static void Convolution<T>(
    CVImage image1,
    CVImage image2,
    ref CVImage imageOut)
    where T : unmanaged,
        INumber<T>
    {
        int kH = (image2.Height - 1) / 2;
        int kW = (image2.Width - 1) / 2;

        Span<T> src = image1.BufferAs<T>();
        Span<T> kernel = image2.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        int width = image1.Width;
        int height = image1.Height;

        int kWidth = image2.Width;
        int simdWidth = Vector<T>.Count;

        for (int y = kH; y < height - kH; y++)
        {
            int rowOut = y * width;

            int x = kW;

            // SIMD main loop
            for (; x <= width - kW - simdWidth; x += simdWidth)
            {
                Vector<T> acc = Vector<T>.Zero;

                for (int dy = -kH; dy <= kH; dy++)
                {
                    int rowIn = (y + dy) * width;
                    int kRow = (dy + kH) * kWidth;

                    for (int dx = -kW; dx <= kW; dx++)
                    {
                        T k = kernel[kRow + dx + kW];

                        int srcIndex = rowIn + x + dx;

                        Vector<T> vec = Vector.LoadUnsafe(ref src[srcIndex]);

                        acc += vec * k;
                    }
                }

                acc.CopyTo(dst.Slice(rowOut + x));
            }

            // scalar tail
            for (; x < width - kW; x++)
            {
                T sum = T.AdditiveIdentity;

                for (int dy = -kH; dy <= kH; dy++)
                {
                    int rowIn = (y + dy) * width;
                    int kRow = (dy + kH) * kWidth;

                    for (int dx = -kW; dx <= kW; dx++)
                    {
                        sum += src[rowIn + x + dx] * kernel[kRow + dx + kW];
                    }
                }

                dst[rowOut + x] = sum;
            }
        }
    }

    public static CVImage ConvolutionX(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat, image1.ChannelFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) ConvolutionX<byte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) ConvolutionX<sbyte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) ConvolutionX<ushort>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) ConvolutionX<short>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) ConvolutionX<uint>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) ConvolutionX<int>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) ConvolutionX<ulong>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) ConvolutionX<long>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) ConvolutionX<float>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) ConvolutionX<double>(image1, image2, ref outImage);

        return outImage;
    }

    public static CVImage ConvolutionY(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat, image1.ChannelFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) ConvolutionY<byte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) ConvolutionY<sbyte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) ConvolutionY<ushort>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) ConvolutionY<short>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) ConvolutionY<uint>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) ConvolutionY<int>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) ConvolutionY<ulong>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) ConvolutionY<long>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) ConvolutionY<float>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) ConvolutionY<double>(image1, image2, ref outImage);

        return outImage;
    }

    public static CVImage Convolution(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat, image1.ChannelFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) Convolution<byte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Convolution<sbyte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Convolution<ushort>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Convolution<short>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Convolution<uint>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Convolution<int>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Convolution<ulong>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Convolution<long>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Convolution<float>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Convolution<double>(image1, image2, ref outImage);

        return outImage;
    }

    public static CVImage MeanBlur(CVImage image, int n)
    {
        CVImage imageOut = SumWindow(image, n, image.DataFormat);
        return CVDivide.Divide(imageOut, n * n);
    }

    public static CVImage GaussianBlur(CVImage image, int n)
    {
        CVImage blurMaskX = CVImage.CreateGaussianMask(n, 1, image.ColorFormat, image.DataFormat);
        CVImage blurMaskY = CVImage.CreateGaussianMask(1, n, image.ColorFormat, image.DataFormat);

        CVImage imageOut = Convolution(image, blurMaskX);
        imageOut = Convolution(imageOut, blurMaskY);

        return imageOut;
    }

    public static CVImage Threshold<T>(CVImage imageIn, T threshold, T min, T max) where T : struct
    {
        return CVMath.Bigger(imageIn, threshold, max, min);
    }

    public static CVImage AdaptiveThresholdMean<T>(CVImage imageIn, T offset, int size) where T : struct, INumber<T>
    {
        CVImage floatImage = CVConvert.ConvertData(imageIn, CVDataFormat.CV_F32);
        CVImage meanBlur = MeanBlur(floatImage, size);
        meanBlur = CVSubtract.Subtract(meanBlur, offset);
        CVImage thresh = CVSmaller.Smaller(floatImage, meanBlur);
        return CVConvert.ConvertData(thresh, imageIn.DataFormat);
    }

    public static CVImage AdaptiveThresholdGauss<T>(CVImage imageIn, T offset, int size) where T : struct
    {
        CVImage floatImage = CVConvert.ConvertData(imageIn, CVDataFormat.CV_F32);
        CVImage meanBlur = GaussianBlur(floatImage, size);
        meanBlur = CVSubtract.Subtract(meanBlur, offset);
        CVImage thresh = CVSmaller.Smaller(floatImage, meanBlur);
        return CVConvert.ConvertData(thresh, imageIn.DataFormat);
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

    public static List<int> Histogram<T>(CVImage imageIn, int bucketCount, out T min, out T bucketSize) where T : struct, INumber<T>
    {
        Console.WriteLine(bucketCount);

        T max = MaxValue<T>(imageIn);
        min = MinValue<T>(imageIn);
        T values = max - min;

        Console.WriteLine(min);
        Console.WriteLine(max);
        Console.WriteLine(values);

        bucketSize = values / (T)Convert.ChangeType(bucketCount, typeof(T));

        Console.WriteLine(bucketSize);

        List<int> histogram = Enumerable.Repeat(0, bucketCount).ToList();
        Span<T> buffer = imageIn.BufferAs<T>();
        for (int i = 0; i < imageIn.Width * imageIn.Height * imageIn.Channels; i++)
        {
            if (bucketSize == T.Zero)
            {
                histogram[0]++;
                continue;
            }

            int bucket = Convert.ToInt32(buffer[i] / bucketSize);

            if (bucket < 0)
                bucket = 0;
            else if (bucket >= bucketCount)
                bucket = bucketCount - 1;

            Console.WriteLine(buffer[i]);
            Console.WriteLine(bucket);


            histogram[bucket]++;
        }

        return histogram;
    }

    public static int OtsuBucket(CVImage imageIn, List<int> histogram)
    {
        int bucketCount = histogram.Count;

        double sumBackground = 0;
        int weightBackground = 0;
        int weightForeground = 0;

        double maxVariance = 0;
        byte threshold = 0;

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
                threshold = (byte)t;
            }
        }

        return threshold;
    }

    public static void OtsuThreshold<T>(CVImage imageIn, int bucketCount, ref CVImage outImage) where T : struct, INumber<T>
    {
        List<int> histogram = Histogram<T>(imageIn, bucketCount, out T min, out T bucketSize);
        int otsuBucket = OtsuBucket(imageIn, histogram);
        T otsuThreshold = min + bucketSize * (T)Convert.ChangeType(otsuBucket, typeof(T));
        outImage = CVSmaller.Smaller<T, T>(imageIn, otsuThreshold);
    }

    public static CVImage OtsuThreshold(CVImage image1, int bucketCount)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat, image1.ChannelFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) OtsuThreshold<byte>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) OtsuThreshold<sbyte>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) OtsuThreshold<ushort>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) OtsuThreshold<short>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) OtsuThreshold<uint>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) OtsuThreshold<int>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) OtsuThreshold<ulong>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) OtsuThreshold<long>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) OtsuThreshold<float>(image1, bucketCount, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) OtsuThreshold<double>(image1, bucketCount, ref outImage);

        return outImage;
    }
}