using System.Numerics;

namespace CVNet;

public static partial class CVMath
{
    private static void notEqual<T>(
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

            var ones = Vector<T>.One;
            var zeros = Vector<T>.Zero;

            for (; i <= count - simdWidth; i += simdWidth)
            {
                Vector<T> vSrc = new(src.Slice(i, simdWidth));
                var mask = Vector.Equals(vSrc, vValue);
                Vector.ConditionalSelect(mask, zeros, ones).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = src[i] != value ? T.One : T.Zero;
        }
    }

    private static void notEqual<T>(
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
            var ones = Vector<T>.One;
            var zeros = Vector<T>.Zero;

            for (; i <= count - simdWidth; i += simdWidth)
            {
                Vector<T> vSrc1 = new(src1.Slice(i, simdWidth));
                Vector<T> vSrc2 = new(src2.Slice(i, simdWidth));
                var mask = Vector.Equals(vSrc1, vSrc2);
                Vector.ConditionalSelect(mask, zeros, ones).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = src1[i] != src2[i] ? T.One : T.Zero;
        }
    }

    public static void NotEqual<T, TV>(
        CVImage imageIn,
        TV value,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct, INumber<TV>
    {
        T valueC = T.CreateChecked(value);

        Span<T> src = imageIn.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        notEqual(src, dst, valueC);
    }

    public static void NotEqual<T, TV>(
        CVImage imageIn,
        TV value,
        int channel,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct, INumber<TV>
    {
        T valueC = T.CreateChecked(value);

        Span<T> src = imageIn.ChannelAs<T>(channel);
        Span<T> dst = imageOut.ChannelAs<T>(channel);

        notEqual(src, dst, valueC);
    }

    public static void NotEqual<T, TV>(
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

            notEqual(src, dst, valueC);
        }
    }

    public static void NotEqual<T>(
        CVImage imageIn1,
        CVImage imageIn2,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T>
    {
        Span<T> src1 = imageIn1.BufferAs<T>();
        Span<T> src2 = imageIn2.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        notEqual(src1, src2, dst);
    }

    public static void NotEqual<T>(
        CVImage imageIn1,
        CVImage imageIn2,
        int channel,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T>
    {
        Span<T> src1 = imageIn1.ChannelAs<T>(channel);
        Span<T> src2 = imageIn2.ChannelAs<T>(channel);
        Span<T> dst = imageOut.ChannelAs<T>(channel);

        notEqual(src1, src2, dst);
    }

    public static void NotEqual<T>(
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

            notEqual(src1, src2, dst);
        }
    }

    public static CVImage NotEqual<T>(CVImage image, T value) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) NotEqual<byte, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) NotEqual<sbyte, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) NotEqual<ushort, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) NotEqual<short, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) NotEqual<uint, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) NotEqual<int, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) NotEqual<ulong, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) NotEqual<long, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) NotEqual<float, T>(image, value, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) NotEqual<double, T>(image, value, ref outImage);

        return outImage;
    }

    public static CVImage NotEqual<T>(CVImage image, T value, int channel) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) NotEqual<byte, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) NotEqual<sbyte, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) NotEqual<ushort, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) NotEqual<short, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) NotEqual<uint, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) NotEqual<int, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) NotEqual<ulong, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) NotEqual<long, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) NotEqual<float, T>(image, value, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) NotEqual<double, T>(image, value, channel, ref outImage);

        return outImage;
    }

    public static CVImage NotEqual<T>(CVImage image, T[] values) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) NotEqual<byte, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) NotEqual<sbyte, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) NotEqual<ushort, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) NotEqual<short, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) NotEqual<uint, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) NotEqual<int, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) NotEqual<ulong, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) NotEqual<long, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) NotEqual<float, T>(image, values, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) NotEqual<double, T>(image, values, ref outImage);

        return outImage;
    }

    public static CVImage NotEqual(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) NotEqual<byte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) NotEqual<sbyte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) NotEqual<ushort>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) NotEqual<short>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) NotEqual<uint>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) NotEqual<int>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) NotEqual<ulong>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) NotEqual<long>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) NotEqual<float>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) NotEqual<double>(image1, image2, ref outImage);

        return outImage;
    }

    public static CVImage NotEqual(CVImage image1, CVImage image2, int channel)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) NotEqual<byte>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) NotEqual<sbyte>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) NotEqual<ushort>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) NotEqual<short>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) NotEqual<uint>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) NotEqual<int>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) NotEqual<ulong>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) NotEqual<long>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) NotEqual<float>(image1, image2, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) NotEqual<double>(image1, image2, channel, ref outImage);

        return outImage;
    }

    public static CVImage NotEqual(CVImage image1, CVImage image2, int[] channels)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) NotEqual<byte>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) NotEqual<sbyte>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) NotEqual<ushort>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) NotEqual<short>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) NotEqual<uint>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) NotEqual<int>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) NotEqual<ulong>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) NotEqual<long>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) NotEqual<float>(image1, image2, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) NotEqual<double>(image1, image2, channels, ref outImage);

        return outImage;
    }

    public static CVImagePyramid NotEqual<T>(CVImagePyramid image, T value) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = NotEqual(image[i], value);

        return outImage;
    }

    public static CVImagePyramid NotEqual<T>(CVImagePyramid image, T value, int channel) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = NotEqual(image[i], value, channel);

        return outImage;
    }

    public static CVImagePyramid NotEqual<T>(CVImagePyramid image, T[] values) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = NotEqual(image[i], values);

        return outImage;
    }

    public static CVImagePyramid NotEqual(CVImagePyramid image1, CVImagePyramid image2)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = NotEqual(image1[i], image2[i]);

        return outImage;
    }

    public static CVImagePyramid NotEqual(CVImagePyramid image1, CVImagePyramid image2, int channel)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = NotEqual(image1[i], image2[i], channel);

        return outImage;
    }

    public static CVImagePyramid NotEqual(CVImagePyramid image1, CVImagePyramid image2, int[] channels)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = NotEqual(image1[i], image2[i], channels);

        return outImage;
    }
}