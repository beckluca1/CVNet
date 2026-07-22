using System.Numerics;

namespace CVNet;

public class CVConvolution
{
    private static void convolutionX<T>(
            CVImage image1,
            CVImage image2,
            ref CVImage imageOut)
            where T : struct, INumber<T>
    {
        int kW = (image2.Width - 1) / 2;

        Span<T> src = image1.BufferAs<T>();
        Span<T> kernel = image2.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        int width = image1.Width;
        int height = image1.Height;

        int planeSize = width * height;

        int simdWidth = Vector<T>.Count;

        // assume kernel is 1D horizontal
        int kWidth = image2.Width;

        for (int c = 0; c < image1.Channels; c++)
        {
            int colorOffset = c * planeSize;

            for (int y = 0; y < height; y++)
            {
                int row = colorOffset + y * width;

                int x = 0;

                for (; x < kW; x++)
                {
                    T sum = T.Zero;

                    for (int dx = -kW; dx <= kW; dx++)
                    {
                        int xIndex = x + dx;

                        if (xIndex < 0) xIndex = 0;
                        else if (xIndex >= width) xIndex = width - 1;

                        sum += src[row + xIndex] * kernel[dx + kW];
                    }

                    dst[row + x] = sum;
                }

                // SIMD main loop
                for (; x <= width - simdWidth - kW; x += simdWidth)
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
                for (; x < width; x++)
                {
                    T sum = T.Zero;

                    for (int dx = -kW; dx <= kW; dx++)
                    {
                        int xIndex = x + dx;

                        if (xIndex < 0) xIndex = 0;
                        else if (xIndex >= width) xIndex = width - 1;

                        sum += src[row + xIndex] * kernel[dx + kW];
                    }

                    dst[row + x] = sum;
                }
            }
        }
    }

    private static void convolutionY<T>(
        CVImage image1,
        CVImage image2,
        ref CVImage imageOut)
        where T : struct, INumber<T>
    {
        int kH = (image2.Height - 1) / 2;
        int width = image1.Width;
        int height = image1.Height;

        int planeSize = width * height;

        Span<T> src = image1.BufferAs<T>();
        Span<T> kernel = image2.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        int simdWidth = Vector<T>.Count;

        for (int c = 0; c < image1.Channels; c++)
        {
            int colorOffset = c * planeSize;

            int y = 0;

            for (; y < kH; y++)
            {
                int rowOut = colorOffset + y * width;

                // scalar tail
                for (int x = 0; x < width; x++)
                {
                    T sum = T.Zero;

                    for (int dy = -kH; dy <= kH; dy++)
                    {
                        int yIndex = y + dy;

                        if (yIndex < 0) yIndex = 0;
                        else if (yIndex >= height) yIndex = height - 1;

                        int rowIn = colorOffset + yIndex * width;
                        sum += src[rowIn + x] * kernel[dy + kH];
                    }

                    dst[rowOut + x] = sum;
                }
            }

            for (; y < height - kH; y++)
            {
                int rowOut = colorOffset + y * width;

                int x = 0;

                // SIMD over X
                for (; x <= width - simdWidth; x += simdWidth)
                {
                    Vector<T> acc = Vector<T>.Zero;

                    for (int dy = -kH; dy <= kH; dy++)
                    {
                        int rowIn = colorOffset + (y + dy) * width;
                        T k = kernel[dy + kH];

                        var v = Vector.LoadUnsafe(ref src[rowIn + x]);

                        acc += v * k;
                    }

                    acc.CopyTo(dst.Slice(rowOut + x));
                }

                // scalar tail
                for (; x < width; x++)
                {
                    T sum = T.Zero;

                    for (int dy = -kH; dy <= kH; dy++)
                    {
                        int yIndex = y + dy;

                        if (yIndex < 0) yIndex = 0;
                        else if (yIndex >= height) yIndex = height - 1;

                        int rowIn = colorOffset + yIndex * width;
                        sum += src[rowIn + x] * kernel[dy + kH];
                    }

                    dst[rowOut + x] = sum;
                }
            }

            for (; y < height; y++)
            {
                int rowOut = colorOffset + y * width;

                // scalar tail
                for (int x = 0; x < width; x++)
                {
                    T sum = T.Zero;

                    for (int dy = -kH; dy <= kH; dy++)
                    {
                        int yIndex = y + dy;

                        if (yIndex < 0) yIndex = 0;
                        else if (yIndex >= height) yIndex = height - 1;

                        int rowIn = colorOffset + yIndex * width;
                        sum += src[rowIn + x] * kernel[dy + kH];
                    }

                    dst[rowOut + x] = sum;
                }
            }
        }
    }

    // Optimized
    private static void convolution2D<T>(
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

        int planeSize = width * height;

        int kWidth = image2.Width;
        int simdWidth = Vector<T>.Count;

        for (int c = 0; c < image1.Channels; c++)
        {
            int colorOffset = c * planeSize;

            for (int y = kH; y < height - kH; y++)
            {
                int rowOut = colorOffset + y * width;

                int x = kW;

                // SIMD main loop
                for (; x <= width - kW - simdWidth; x += simdWidth)
                {
                    Vector<T> acc = Vector<T>.Zero;

                    for (int dy = -kH; dy <= kH; dy++)
                    {
                        int rowIn = colorOffset + (y + dy) * width;
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
                        int rowIn = colorOffset + (y + dy) * width;
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
    }

    public static CVImage ConvolutionX(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) convolutionX<byte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) convolutionX<sbyte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) convolutionX<ushort>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) convolutionX<short>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) convolutionX<uint>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) convolutionX<int>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) convolutionX<ulong>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) convolutionX<long>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) convolutionX<float>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) convolutionX<double>(image1, image2, ref outImage);

        return outImage;
    }

    public static CVImage ConvolutionY(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) convolutionY<byte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) convolutionY<sbyte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) convolutionY<ushort>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) convolutionY<short>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) convolutionY<uint>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) convolutionY<int>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) convolutionY<ulong>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) convolutionY<long>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) convolutionY<float>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) convolutionY<double>(image1, image2, ref outImage);

        return outImage;
    }

    public static CVImage Convolution2D(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) convolution2D<byte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) convolution2D<sbyte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) convolution2D<ushort>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) convolution2D<short>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) convolution2D<uint>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) convolution2D<int>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) convolution2D<ulong>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) convolution2D<long>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) convolution2D<float>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) convolution2D<double>(image1, image2, ref outImage);

        return outImage;
    }
}