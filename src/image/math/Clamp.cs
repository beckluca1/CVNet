using System.Numerics;

namespace CVNet;

public static partial class CVMath
{
    private static void clamp<T>(
        Span<T> src,
        Span<T> dst,
        T valueMin,
        T valueMax)
        where T : unmanaged, INumber<T>
    {
        int count = src.Length;

        int simdWidth = Vector<T>.Count;
        int i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            Vector<T> vValueMin = new(valueMin);
            Vector<T> vValueMax = new(valueMax);

            for (; i <= count - simdWidth; i += simdWidth)
            {
                Vector<T> vSrc = new(src.Slice(i, simdWidth));
                Vector.Min(Vector.Max(vSrc, vValueMin), vValueMax).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = T.Min(T.Max(src[i], valueMin), valueMax);
        }
    }

    private static void clamp<T>(
        Span<T> src1,
        Span<T> src2,
        Span<T> src3,
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
                Vector<T> vSrc3 = new(src3.Slice(i, simdWidth));
                Vector.Min(Vector.Max(vSrc1, vSrc2), vSrc3).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = T.Min(T.Max(src1[i], src2[i]), src3[i]);
        }
    }

    public static void Clamp<T, TV>(
        CVImage imageIn,
        TV valueMin,
        TV valueMax,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct, INumber<TV>
    {
        T valueMinC = T.CreateChecked(valueMin);
        T valueMaxC = T.CreateChecked(valueMax);

        Span<T> src = imageIn.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        clamp(src, dst, valueMinC, valueMaxC);
    }

    public static void Clamp<T, TV>(
        CVImage imageIn,
        TV valueMin,
        TV valueMax,
        int channel,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct, INumber<TV>
    {
        T valueMinC = T.CreateChecked(valueMin);
        T valueMaxC = T.CreateChecked(valueMax);

        Span<T> src = imageIn.ChannelAs<T>(channel);
        Span<T> dst = imageOut.ChannelAs<T>(channel);

        clamp(src, dst, valueMinC, valueMaxC);
    }

    public static void Clamp<T, TV>(
        CVImage imageIn,
        TV[] valuesMin,
        TV[] valuesMax,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct, INumber<TV>
    {
        for (int channel = 0; channel < imageIn.Channels; channel++)
        {
            T valueMinC = T.CreateChecked(valuesMin[channel]);
            T valueMaxC = T.CreateChecked(valuesMax[channel]);

            Span<T> src = imageIn.ChannelAs<T>(channel);
            Span<T> dst = imageOut.ChannelAs<T>(channel);

            clamp(src, dst, valueMinC, valueMaxC);
        }
    }

    public static void Clamp<T>(
        CVImage imageIn1,
        CVImage imageIn2,
        CVImage imageIn3,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T>
    {
        Span<T> src1 = imageIn1.BufferAs<T>();
        Span<T> src2 = imageIn2.BufferAs<T>();
        Span<T> src3 = imageIn3.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        clamp(src1, src2, src3, dst);
    }

    public static void Clamp<T>(
        CVImage imageIn1,
        CVImage imageIn2,
        CVImage imageIn3,
        int channel,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T>
    {
        Span<T> src1 = imageIn1.ChannelAs<T>(channel);
        Span<T> src2 = imageIn2.ChannelAs<T>(channel);
        Span<T> src3 = imageIn3.ChannelAs<T>(channel);
        Span<T> dst = imageOut.ChannelAs<T>(channel);

        clamp(src1, src2, src3, dst);
    }

    public static void Clamp<T>(
        CVImage imageIn1,
        CVImage imageIn2,
        CVImage imageIn3,
        int[] channels,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T>
    {
        foreach (int channel in channels)
        {
            Span<T> src1 = imageIn1.ChannelAs<T>(channel);
            Span<T> src2 = imageIn2.ChannelAs<T>(channel);
            Span<T> src3 = imageIn3.ChannelAs<T>(channel);
            Span<T> dst = imageOut.ChannelAs<T>(channel);

            clamp(src1, src2, src3, dst);
        }
    }

    public static CVImage Clamp<T>(CVImage image, T valueMin, T valueMax) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Clamp<byte, T>(image, valueMin, valueMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Clamp<sbyte, T>(image, valueMin, valueMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Clamp<ushort, T>(image, valueMin, valueMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Clamp<short, T>(image, valueMin, valueMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Clamp<uint, T>(image, valueMin, valueMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Clamp<int, T>(image, valueMin, valueMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Clamp<ulong, T>(image, valueMin, valueMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Clamp<long, T>(image, valueMin, valueMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Clamp<float, T>(image, valueMin, valueMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Clamp<double, T>(image, valueMin, valueMax, ref outImage);

        return outImage;
    }

    public static CVImage Clamp<T>(CVImage image, T valueMin, T valueMax, int channel) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Clamp<byte, T>(image, valueMin, valueMax, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Clamp<sbyte, T>(image, valueMin, valueMax, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Clamp<ushort, T>(image, valueMin, valueMax, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Clamp<short, T>(image, valueMin, valueMax, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Clamp<uint, T>(image, valueMin, valueMax, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Clamp<int, T>(image, valueMin, valueMax, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Clamp<ulong, T>(image, valueMin, valueMax, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Clamp<long, T>(image, valueMin, valueMax, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Clamp<float, T>(image, valueMin, valueMax, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Clamp<double, T>(image, valueMin, valueMax, channel, ref outImage);

        return outImage;
    }

    public static CVImage Clamp<T>(CVImage image, T[] valuesMin, T[] valuesMax) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Clamp<byte, T>(image, valuesMin, valuesMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Clamp<sbyte, T>(image, valuesMin, valuesMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Clamp<ushort, T>(image, valuesMin, valuesMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Clamp<short, T>(image, valuesMin, valuesMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Clamp<uint, T>(image, valuesMin, valuesMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Clamp<int, T>(image, valuesMin, valuesMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Clamp<ulong, T>(image, valuesMin, valuesMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Clamp<long, T>(image, valuesMin, valuesMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Clamp<float, T>(image, valuesMin, valuesMax, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Clamp<double, T>(image, valuesMin, valuesMax, ref outImage);

        return outImage;
    }

    public static CVImage Clamp(CVImage image1, CVImage image2, CVImage image3)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) Clamp<byte>(image1, image2, image3, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Clamp<sbyte>(image1, image2, image3, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Clamp<ushort>(image1, image2, image3, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Clamp<short>(image1, image2, image3, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Clamp<uint>(image1, image2, image3, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Clamp<int>(image1, image2, image3, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Clamp<ulong>(image1, image2, image3, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Clamp<long>(image1, image2, image3, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Clamp<float>(image1, image2, image3, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Clamp<double>(image1, image2, image3, ref outImage);

        return outImage;
    }

    public static CVImage Clamp(CVImage image1, CVImage image2, CVImage image3, int channel)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) Clamp<byte>(image1, image2, image3, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Clamp<sbyte>(image1, image2, image3, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Clamp<ushort>(image1, image2, image3, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Clamp<short>(image1, image2, image3, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Clamp<uint>(image1, image2, image3, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Clamp<int>(image1, image2, image3, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Clamp<ulong>(image1, image2, image3, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Clamp<long>(image1, image2, image3, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Clamp<float>(image1, image2, image3, channel, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Clamp<double>(image1, image2, image3, channel, ref outImage);

        return outImage;
    }

    public static CVImage Clamp(CVImage image1, CVImage image2, CVImage image3, int[] channels)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) Clamp<byte>(image1, image2, image3, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Clamp<sbyte>(image1, image2, image3, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Clamp<ushort>(image1, image2, image3, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Clamp<short>(image1, image2, image3, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Clamp<uint>(image1, image2, image3, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Clamp<int>(image1, image2, image3, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Clamp<ulong>(image1, image2, image3, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Clamp<long>(image1, image2, image3, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Clamp<float>(image1, image2, image3, channels, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Clamp<double>(image1, image2, image3, channels, ref outImage);

        return outImage;
    }

    public static CVImagePyramid Clamp<T>(CVImagePyramid image, T valueMin, T valueMax) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Clamp(image[i], valueMin, valueMax);

        return outImage;
    }

    public static CVImagePyramid Clamp<T>(CVImagePyramid image, T valueMin, T valueMax, int channel) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Clamp(image[i], valueMin, valueMax, channel);

        return outImage;
    }

    public static CVImagePyramid Clamp<T>(CVImagePyramid image, T[] valuesMin, T[] valuesMax) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Clamp(image[i], valuesMin, valuesMax);

        return outImage;
    }

    public static CVImagePyramid Clamp(CVImagePyramid image1, CVImagePyramid image2, CVImagePyramid image3)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = Clamp(image1[i], image2[i], image3[i]);

        return outImage;
    }

    public static CVImagePyramid Clamp(CVImagePyramid image1, CVImagePyramid image2, CVImagePyramid image3, int channel)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = Clamp(image1[i], image2[i], image3[i], channel);

        return outImage;
    }

    public static CVImagePyramid Clamp(CVImagePyramid image1, CVImagePyramid image2, CVImagePyramid image3, int[] channels)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = Clamp(image1[i], image2[i], image3[i], channels);

        return outImage;
    }
}