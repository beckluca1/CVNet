using System.Numerics;

namespace CVNet;

public static partial class CVMath
{
    private static void divide<T>(
        Span<T> src,
        Span<T> dst,
        T value)
        where T : unmanaged, INumber<T>
    {
        int count = src.Length;

        int simdWidth = Vector<T>.Count;
        int i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            Vector<T> vValue = new(value);

            for (; i <= count - simdWidth; i += simdWidth)
            {
                Vector<T> vSrc = new(src.Slice(i, simdWidth));
                (vSrc / vValue).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = src[i] / value;
        }
    }

    private static void divide<T>(
        T value,
        Span<T> dst,
        Span<T> src)
        where T : unmanaged, INumber<T>
    {
        int count = src.Length;

        int simdWidth = Vector<T>.Count;
        int i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            Vector<T> vValue = new(value);

            for (; i <= count - simdWidth; i += simdWidth)
            {
                Vector<T> vSrc = new(src.Slice(i, simdWidth));
                (vValue / vSrc).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = value / src[i];
        }
    }

    private static void divide<T>(
        Span<T> src1,
        Span<T> src2,
        Span<T> dst)
        where T : unmanaged, INumber<T>
    {
        int count = src1.Length;

        int simdWidth = Vector<T>.Count;
        int i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= count - simdWidth; i += simdWidth)
            {
                Vector<T> vSrc1 = new(src1.Slice(i, simdWidth));
                Vector<T> vSrc2 = new(src2.Slice(i, simdWidth));
                (vSrc1 / vSrc2).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = src1[i] / src2[i];
        }
    }

    public static void Divide<T, TV>(
        CVImage imageIn,
        TV value,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct, INumber<TV>
    {
        T valueC = T.CreateChecked(value);

        Span<T> src = imageIn.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        divide(src, dst, valueC);
    }

    public static void Divide<T, TV>(
        CVImage imageIn,
        TV value,
        int channel,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct, INumber<TV>
    {
        T valueC = T.CreateChecked(value);

        Span<T> src = imageIn.ChannelAs<T>(channel);
        Span<T> dst = imageOut.ChannelAs<T>(channel);

        divide(src, dst, valueC);
    }

    public static void Divide<T, TV>(
        CVImage imageIn,
        TV[] values,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct, INumber<TV>
    {
        for (int channel = 0; channel < imageIn.Channels; channel++)
        {
            T valueC = T.CreateChecked(values[channel]);

            Span<T> src = imageIn.ChannelAs<T>(channel);
            Span<T> dst = imageOut.ChannelAs<T>(channel);

            divide(src, dst, valueC);
        }
    }

    public static void Divide<T, TV>(
            TV value,
            CVImage imageIn,
            ref CVImage imageOut)
            where T : unmanaged, INumber<T> where TV : struct, INumber<TV>
    {
        T valueC = T.CreateChecked(value);

        Span<T> src = imageIn.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        divide(valueC, dst, src);
    }

    public static void Divide<T, TV>(
        TV value,
        CVImage imageIn,
        int channel,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct, INumber<TV>
    {
        T valueC = T.CreateChecked(value);

        Span<T> src = imageIn.ChannelAs<T>(channel);
        Span<T> dst = imageOut.ChannelAs<T>(channel);

        divide(valueC, dst, src);
    }

    public static void Divide<T, TV>(
        TV[] values,
        CVImage imageIn,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct, INumber<TV>
    {
        for (int channel = 0; channel < imageIn.Channels; channel++)
        {
            T valueC = T.CreateChecked(values[channel]);

            Span<T> src = imageIn.ChannelAs<T>(channel);
            Span<T> dst = imageOut.ChannelAs<T>(channel);

            divide(valueC, dst, src);
        }
    }

    public static void Divide<T>(
        CVImage imageIn1,
        CVImage imageIn2,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T>
    {
        Span<T> src1 = imageIn1.BufferAs<T>();
        Span<T> src2 = imageIn2.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        divide(src1, src2, dst);
    }

    public static void Divide<T>(
        CVImage imageIn1,
        CVImage imageIn2,
        int channel,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T>
    {
        Span<T> src1 = imageIn1.ChannelAs<T>(channel);
        Span<T> src2 = imageIn2.ChannelAs<T>(channel);
        Span<T> dst = imageOut.ChannelAs<T>(channel);

        divide(src1, src2, dst);
    }

    public static void Divide<T>(
        CVImage imageIn1,
        CVImage imageIn2,
        int[] channels,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T>
    {
        foreach (int channel in channels)
        {
            Span<T> src1 = imageIn1.ChannelAs<T>(channel);
            Span<T> src2 = imageIn2.ChannelAs<T>(channel);
            Span<T> dst = imageOut.ChannelAs<T>(channel);

            divide(src1, src2, dst);
        }
    }

    public static CVImage Divide<T>(CVImage image, T value) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Divide<byte, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Divide<sbyte, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Divide<ushort, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Divide<short, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Divide<uint, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Divide<int, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Divide<ulong, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Divide<long, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Divide<float, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Divide<double, T>(image, value, ref outImage);

        return outImage;
    }

    public static CVImage Divide<T>(CVImage image, T value, int channel) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Divide<byte, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Divide<sbyte, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Divide<ushort, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Divide<short, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Divide<uint, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Divide<int, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Divide<ulong, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Divide<long, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Divide<float, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Divide<double, T>(image, value, channel, ref outImage);

        return outImage;
    }

    public static CVImage Divide<T>(CVImage image, T[] values) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Divide<byte, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Divide<sbyte, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Divide<ushort, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Divide<short, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Divide<uint, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Divide<int, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Divide<ulong, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Divide<long, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Divide<float, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Divide<double, T>(image, values, ref outImage);

        return outImage;
    }

    public static CVImage Divide<T>(T value, CVImage image) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Divide<byte, T>(value, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Divide<sbyte, T>(value, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Divide<ushort, T>(value, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Divide<short, T>(value, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Divide<uint, T>(value, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Divide<int, T>(value, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Divide<ulong, T>(value, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Divide<long, T>(value, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Divide<float, T>(value, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Divide<double, T>(value, image, ref outImage);

        return outImage;
    }

    public static CVImage Divide<T>(T value, CVImage image, int channel) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Divide<byte, T>(value, image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Divide<sbyte, T>(value, image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Divide<ushort, T>(value, image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Divide<short, T>(value, image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Divide<uint, T>(value, image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Divide<int, T>(value, image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Divide<ulong, T>(value, image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Divide<long, T>(value, image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Divide<float, T>(value, image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Divide<double, T>(value, image, channel, ref outImage);

        return outImage;
    }

    public static CVImage Divide<T>(T[] values, CVImage image) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Divide<byte, T>(values, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Divide<sbyte, T>(values, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Divide<ushort, T>(values, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Divide<short, T>(values, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Divide<uint, T>(values, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Divide<int, T>(values, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Divide<ulong, T>(values, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Divide<long, T>(values, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Divide<float, T>(values, image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Divide<double, T>(values, image, ref outImage);

        return outImage;
    }

    public static CVImage Divide(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) Divide<byte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Divide<sbyte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Divide<ushort>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Divide<short>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Divide<uint>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Divide<int>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Divide<ulong>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Divide<long>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Divide<float>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Divide<double>(image1, image2, ref outImage);

        return outImage;
    }

    public static CVImage Divide(CVImage image1, CVImage image2, int channel)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) Divide<byte>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Divide<sbyte>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Divide<ushort>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Divide<short>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Divide<uint>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Divide<int>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Divide<ulong>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Divide<long>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Divide<float>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Divide<double>(image1, image2, channel, ref outImage);

        return outImage;
    }

    public static CVImage Divide(CVImage image1, CVImage image2, int[] channels)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) Divide<byte>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Divide<sbyte>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Divide<ushort>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Divide<short>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Divide<uint>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Divide<int>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Divide<ulong>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Divide<long>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Divide<float>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Divide<double>(image1, image2, channels, ref outImage);

        return outImage;
    }

    public static CVImagePyramid Divide<T>(CVImagePyramid image, T value) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Divide(image[i], value);

        return outImage;
    }

    public static CVImagePyramid Divide<T>(CVImagePyramid image, T value, int channel) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Divide(image[i], value, channel);

        return outImage;
    }

    public static CVImagePyramid Divide<T>(CVImagePyramid image, T[] values) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Divide(image[i], values);

        return outImage;
    }

    public static CVImagePyramid Divide<T>(T value, CVImagePyramid image) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Divide(value, image[i]);

        return outImage;
    }

    public static CVImagePyramid Divide<T>(T value, CVImagePyramid image, int channel) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Divide(value, image[i], channel);

        return outImage;
    }

    public static CVImagePyramid Divide<T>(T[] values, CVImagePyramid image) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Divide(values, image[i]);

        return outImage;
    }

    public static CVImagePyramid Divide(CVImagePyramid image1, CVImagePyramid image2)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = Divide(image1[i], image2[i]);

        return outImage;
    }

    public static CVImagePyramid Divide(CVImagePyramid image1, CVImagePyramid image2, int channel)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = Divide(image1[i], image2[i], channel);

        return outImage;
    }

    public static CVImagePyramid Divide(CVImagePyramid image1, CVImagePyramid image2, int[] channels)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = Divide(image1[i], image2[i], channels);

        return outImage;
    }
}