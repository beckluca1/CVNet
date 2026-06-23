using System.Numerics;

namespace CVNet;

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

    public static CVImage IntegralImage(CVImage image1)
    {
        CVImage outImage = CVImage.Create(image1.Width + 1, image1.Height + 1, image1.ColorFormat, image1.DataFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) IntegralImage<byte>(image1, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) IntegralImage<sbyte>(image1, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) IntegralImage<ushort>(image1, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) IntegralImage<short>(image1, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) IntegralImage<uint>(image1, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) IntegralImage<int>(image1, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) IntegralImage<ulong>(image1, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) IntegralImage<long>(image1, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) IntegralImage<float>(image1, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) IntegralImage<double>(image1, ref outImage);

        return outImage;
    }

    public static void SumWindow<T>(
    CVImage image1,
    int size,
    ref CVImage imageOut)
    where T : struct, INumber<T>
    {
        CVImage integralImage = IntegralImage(image1);

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

                    dst[dstRow + x] =
                        src[rowCD + x1]
                      - src[rowAB + x1]
                      - src[rowCD + x0]
                      + src[rowAB + x0];
                }
            }
        }
    }

    public static CVImage SumWindow(CVImage image1, int size)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) SumWindow<byte>(image1, size, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) SumWindow<sbyte>(image1, size, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) SumWindow<ushort>(image1, size, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) SumWindow<short>(image1, size, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) SumWindow<uint>(image1, size, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) SumWindow<int>(image1, size, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) SumWindow<ulong>(image1, size, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) SumWindow<long>(image1, size, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) SumWindow<float>(image1, size, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) SumWindow<double>(image1, size, ref outImage);

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
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

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
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

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
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

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
        CVImage imageOut = SumWindow(image, n);
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

    public static CVImage AdaptiveThresholdGauss<T>(CVImage imageIn, T max, T min, T offset, int size) where T : struct
    {
        CVImage gaussianMean = GaussianBlur(imageIn, size);
        CVImage imageOffset = CVMath.Add(imageIn, offset);
        return CVMath.Bigger(imageOffset, gaussianMean, max, min);
    }
}