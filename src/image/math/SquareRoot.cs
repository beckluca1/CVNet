using System.Numerics;

namespace CVNet;

public static partial class CVMath
{
    private static void squareRoot<T>(
        Span<T> src,
        Span<T> dst)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int count = src.Length;

        int simdWidth = Vector<T>.Count;
        int i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= count - simdWidth; i += simdWidth)
            {
                Vector<T> vSrc = new(src.Slice(i, simdWidth));
                Vector.SquareRoot(vSrc).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = T.Sqrt(src[i]);
        }
    }

    public static void SquareRoot<T>(
        CVImage imageIn,
        ref CVImage imageOut)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Span<T> src = imageIn.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        squareRoot(src, dst);
    }

    public static void SquareRoot<T>(
        CVImage imageIn,
        int channel,
        ref CVImage imageOut)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Span<T> src = imageIn.ChannelAs<T>(channel);
        Span<T> dst = imageOut.ChannelAs<T>(channel);

        squareRoot(src, dst);
    }

    public static void SquareRoot<T>(
        CVImage imageIn,
        int[] channels,
        ref CVImage imageOut)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        foreach (int channel in channels)
        {
            Span<T> src = imageIn.ChannelAs<T>(channel);
            Span<T> dst = imageOut.ChannelAs<T>(channel);

            squareRoot(src, dst);
        }
    }

    public static CVImage SquareRoot(CVImage image)
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_F32) SquareRoot<float>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) SquareRoot<double>(image, ref outImage);

        return outImage;
    }

    public static CVImage SquareRoot(CVImage image, int channel)
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_F32) SquareRoot<float>(image, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) SquareRoot<double>(image, channel, ref outImage);

        return outImage;
    }

    public static CVImage SquareRoot(CVImage image, int[] channels)
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_F32) SquareRoot<float>(image, channels, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) SquareRoot<double>(image, channels, ref outImage);

        return outImage;
    }

    public static CVImagePyramid SquareRoot(CVImagePyramid image)
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = SquareRoot(image[i]);

        return outImage;
    }

    public static CVImagePyramid SquareRoot<T>(CVImagePyramid image, int channel)
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = SquareRoot(image[i], channel);

        return outImage;
    }


    public static CVImagePyramid SquareRoot<T>(CVImagePyramid image, int[] channels)
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = SquareRoot(image[i], channels);

        return outImage;
    }
}