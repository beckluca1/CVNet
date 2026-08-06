using System.Numerics;

namespace CVNet;

public static partial class CVMath
{
    private static void add<T>(
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
                (vSrc + vValue).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = src[i] + value;
        }
    }

    private static void add<T>(
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
                (vSrc1 + vSrc2).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = src1[i] + src2[i];
        }
    }

    public static void Add<T, TV>(
        CVImage imageIn,
        TV value,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct, INumber<TV>
    {
        T valueC = T.CreateChecked(value);

        Span<T> src = imageIn.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        add(src, dst, valueC);
    }

    public static void Add<T, TV>(
        CVImage imageIn,
        TV value,
        int channel,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct, INumber<TV>
    {
        T valueC = T.CreateChecked(value);

        Span<T> src = imageIn.ChannelAs<T>(channel);
        Span<T> dst = imageOut.ChannelAs<T>(channel);

        add(src, dst, valueC);
    }

    public static void Add<T, TV>(
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

            add(src, dst, valueC);
        }
    }

    public static void Add<T>(
        CVImage imageIn1,
        CVImage imageIn2,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T>
    {
        Span<T> src1 = imageIn1.BufferAs<T>();
        Span<T> src2 = imageIn2.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        add(src1, src2, dst);
    }

    public static void Add<T>(
        CVImage imageIn1,
        CVImage imageIn2,
        int channel,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T>
    {
        Span<T> src1 = imageIn1.ChannelAs<T>(channel);
        Span<T> src2 = imageIn2.ChannelAs<T>(channel);
        Span<T> dst = imageOut.ChannelAs<T>(channel);

        add(src1, src2, dst);
    }

    public static void Add<T>(
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

            add(src1, src2, dst);
        }
    }

    public static CVImage Add<T>(CVImage image, T value) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Add<byte, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Add<sbyte, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Add<ushort, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Add<short, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Add<uint, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Add<int, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Add<ulong, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Add<long, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Add<float, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Add<double, T>(image, value, ref outImage);

        return outImage;
    }

    public static CVImage Add<T>(CVImage image, T value, int channel) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Add<byte, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Add<sbyte, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Add<ushort, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Add<short, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Add<uint, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Add<int, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Add<ulong, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Add<long, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Add<float, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Add<double, T>(image, value, channel, ref outImage);

        return outImage;
    }

    public static CVImage Add<T>(CVImage image, T[] values) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Add<byte, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Add<sbyte, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Add<ushort, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Add<short, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Add<uint, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Add<int, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Add<ulong, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Add<long, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Add<float, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Add<double, T>(image, values, ref outImage);

        return outImage;
    }

    public static CVImage Add(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) Add<byte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Add<sbyte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Add<ushort>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Add<short>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Add<uint>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Add<int>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Add<ulong>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Add<long>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Add<float>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Add<double>(image1, image2, ref outImage);

        return outImage;
    }

    public static CVImage Add(CVImage image1, CVImage image2, int channel)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) Add<byte>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Add<sbyte>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Add<ushort>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Add<short>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Add<uint>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Add<int>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Add<ulong>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Add<long>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Add<float>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Add<double>(image1, image2, channel, ref outImage);

        return outImage;
    }

    public static CVImage Add(CVImage image1, CVImage image2, int[] channels)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) Add<byte>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Add<sbyte>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Add<ushort>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Add<short>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Add<uint>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Add<int>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Add<ulong>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Add<long>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Add<float>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Add<double>(image1, image2, channels, ref outImage);

        return outImage;
    }

    public static CVImagePyramid Add<T>(CVImagePyramid image, T value) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Add(image[i], value);

        return outImage;
    }

    public static CVImagePyramid Add<T>(CVImagePyramid image, T value, int channel) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Add(image[i], value, channel);

        return outImage;
    }

    public static CVImagePyramid Add<T>(CVImagePyramid image, T[] values) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Add(image[i], values);

        return outImage;
    }

    public static CVImagePyramid Add(CVImagePyramid image1, CVImagePyramid image2)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = Add(image1[i], image2[i]);

        return outImage;
    }

    public static CVImagePyramid Add(CVImagePyramid image1, CVImagePyramid image2, int channel)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = Add(image1[i], image2[i], channel);

        return outImage;
    }

    public static CVImagePyramid Add(CVImagePyramid image1, CVImagePyramid image2, int[] channels)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = Add(image1[i], image2[i], channels);

        return outImage;
    }
}