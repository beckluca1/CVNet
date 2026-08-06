using System.Numerics;

namespace CVNet;

public static partial class CVMath
{
    private static void abs<T>(
        Span<T> src,
        Span<T> dst)
        where T : unmanaged, INumber<T>
    {
        int count = src.Length;

        int simdWidth = Vector<T>.Count;
        int i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= count - simdWidth; i += simdWidth)
            {
                Vector<T> vSrc = new(src.Slice(i, simdWidth));
                Vector.Abs(vSrc).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = T.Abs(src[i]);
        }
    }

    public static void Abs<T>(
        CVImage imageIn,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T>
    {
        Span<T> src = imageIn.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        abs(src, dst);
    }

    public static void Abs<T>(
        CVImage imageIn,
        int channel,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T>
    {
        Span<T> src = imageIn.ChannelAs<T>(channel);
        Span<T> dst = imageOut.ChannelAs<T>(channel);

        abs(src, dst);
    }

    public static void Abs<T>(
        CVImage imageIn,
        int[] channels,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T>
    {
        foreach (int channel in channels)
        {
            Span<T> src = imageIn.ChannelAs<T>(channel);
            Span<T> dst = imageOut.ChannelAs<T>(channel);

            abs(src, dst);
        }
    }

    public static CVImage Abs(CVImage image)
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Abs<byte>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Abs<sbyte>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Abs<ushort>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Abs<short>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Abs<uint>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Abs<int>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Abs<ulong>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Abs<long>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Abs<float>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Abs<double>(image, ref outImage);

        return outImage;
    }

    public static CVImage Abs(CVImage image, int channel)
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Abs<byte>(image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Abs<sbyte>(image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Abs<ushort>(image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Abs<short>(image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Abs<uint>(image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Abs<int>(image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Abs<ulong>(image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Abs<long>(image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Abs<float>(image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Abs<double>(image, channel, ref outImage);

        return outImage;
    }

    public static CVImage Abs(CVImage image, int[] channels)
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Abs<byte>(image, channels, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Abs<sbyte>(image, channels, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Abs<ushort>(image, channels, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Abs<short>(image, channels, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Abs<uint>(image, channels, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Abs<int>(image, channels, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Abs<ulong>(image, channels, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Abs<long>(image, channels, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Abs<float>(image, channels, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Abs<double>(image, channels, ref outImage);

        return outImage;
    }

    public static CVImagePyramid Abs(CVImagePyramid image)
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Abs(image[i]);

        return outImage;
    }

    public static CVImagePyramid Abs<T>(CVImagePyramid image, int channel)
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Abs(image[i], channel);

        return outImage;
    }


    public static CVImagePyramid Abs<T>(CVImagePyramid image, int[] channels)
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Abs(image[i], channels);

        return outImage;
    }
}