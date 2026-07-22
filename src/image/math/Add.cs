using System.Numerics;

namespace CVNet;

public static class CVAdd
{
    public static void Add<T, TV>(
        CVImage imageIn,
        TV value,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct, INumber<TV>
    {
        T valueC = T.CreateChecked(value);

        Span<T> src = imageIn.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        int count = src.Length;

        int simdWidth = Vector<T>.Count;
        int i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            Vector<T> vValue = new(valueC);

            for (; i <= count - simdWidth; i += simdWidth)
            {
                Vector<T> vSrc = new(src.Slice(i, simdWidth));
                (vSrc + vValue).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = src[i] + valueC;
        }
    }

    public static void Add<T, TV>(
        CVImage imageIn,
        TV[] values,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct, INumber<TV>
    {
        T[] valuesC = new T[values.Length];
        for (int i = 0; i < valuesC.Length; i++) valuesC[i] = T.CreateChecked(values[i]);

        int planeSize = imageIn.Width * imageIn.Height;

        Span<T> src = imageIn.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        int simdWidth = Vector<T>.Count;

        for (int c = 0; c < imageIn.Channels; c++)
        {
            int baseIdx = c * planeSize;

            Vector<T> vValue = new(valuesC[c]);

            int i = 0;

            if (Vector.IsHardwareAccelerated)
            {
                for (; i <= planeSize - simdWidth; i += simdWidth)
                {
                    int idx = baseIdx + i;

                    Vector<T> vSrc =
                        new(src.Slice(idx, simdWidth));

                    (vSrc + vValue)
                        .CopyTo(dst.Slice(idx, simdWidth));
                }
            }

            for (; i < planeSize; i++)
            {
                int idx = baseIdx + i;
                dst[idx] = src[idx] + valuesC[c];
            }
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

        int count = src1.Length;

        int simdWidth = Vector<T>.Count;
        int i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= count - simdWidth; i += simdWidth)
            {
                Vector<T> v1 = new(src1.Slice(i, simdWidth));
                Vector<T> v2 = new(src2.Slice(i, simdWidth));

                (v1 + v2).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = src1[i] + src2[i];
        }
    }

    public static CVImage Add<T>(CVImage image, T arg1) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Add<byte, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Add<sbyte, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Add<ushort, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Add<short, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Add<uint, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Add<int, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Add<ulong, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Add<long, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Add<float, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Add<double, T>(image, arg1, ref outImage);

        return outImage;
    }

    public static CVImage Add<T>(CVImage image, T[] arg1) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Add<byte, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Add<sbyte, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Add<ushort, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Add<short, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Add<uint, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Add<int, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Add<ulong, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Add<long, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Add<float, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Add<double, T>(image, arg1, ref outImage);

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

        int count = src.Length;

        int simdWidth = Vector<T>.Count;
        int i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            Vector<T> vValue = new(valueC);

            for (; i <= count - simdWidth; i += simdWidth)
            {
                Vector<T> vSrc = new(src.Slice(i, simdWidth));
                (vSrc + vValue).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = src[i] + valueC;
        }
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

        int count = src1.Length;

        int simdWidth = Vector<T>.Count;
        int i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= count - simdWidth; i += simdWidth)
            {
                Vector<T> v1 = new(src1.Slice(i, simdWidth));
                Vector<T> v2 = new(src2.Slice(i, simdWidth));

                (v1 + v2).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = src1[i] + src2[i];
        }
    }

    public static CVImage Add<T>(CVImage image, T arg1, int channel) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Add<byte, T>(image, arg1, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Add<sbyte, T>(image, arg1, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Add<ushort, T>(image, arg1, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Add<short, T>(image, arg1, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Add<uint, T>(image, arg1, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Add<int, T>(image, arg1, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Add<ulong, T>(image, arg1, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Add<long, T>(image, arg1, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Add<float, T>(image, arg1, channel, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Add<double, T>(image, arg1, channel, ref outImage);

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

    public static CVImagePyramid Add<T>(CVImagePyramid image, T arg1) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Add(image[i], arg1);

        return outImage;
    }

    public static CVImagePyramid Add<T>(CVImagePyramid image, T[] arg1) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Add(image[i], arg1);

        return outImage;
    }

    public static CVImagePyramid Add(CVImagePyramid image1, CVImagePyramid image2)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = Add(image1[i], image2[i]);

        return outImage;
    }

    public static CVImagePyramid Add<T>(CVImagePyramid image, T arg1, int channel) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Add(image[i], arg1, channel);

        return outImage;
    }

    public static CVImagePyramid Add(CVImagePyramid image1, CVImagePyramid image2, int channel)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = Add(image1[i], image2[i], channel);

        return outImage;
    }
}