using System.Numerics;

namespace CVNet;

public static class CVMin
{
    public static void Min<T, TV>(
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
                Vector.Min(vSrc, vValue).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = T.Min(src[i], valueC);
        }
    }

    public static void Min<T, TV>(
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

                    Vector.Min(vSrc, vValue).CopyTo(dst.Slice(idx, simdWidth));
                }
            }

            for (; i < planeSize; i++)
            {
                int idx = baseIdx + i;
                dst[idx] = T.Min(src[idx], valuesC[c]);
            }
        }
    }

    public static void Min<T>(
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

                Vector.Min(v1, v2).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = T.Min(src1[i], src2[i]);
        }
    }

    public static CVImage Min<T>(CVImage image, T arg1) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Min<byte, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Min<sbyte, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Min<ushort, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Min<short, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Min<uint, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Min<int, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Min<ulong, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Min<long, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Min<float, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Min<double, T>(image, arg1, ref outImage);

        return outImage;
    }

    public static CVImage Min<T>(CVImage image, T[] arg1) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Min<byte, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Min<sbyte, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Min<ushort, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Min<short, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Min<uint, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Min<int, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Min<ulong, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Min<long, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Min<float, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Min<double, T>(image, arg1, ref outImage);

        return outImage;
    }

    public static CVImage Min(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) Min<byte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Min<sbyte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Min<ushort>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Min<short>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Min<uint>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Min<int>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Min<ulong>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Min<long>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Min<float>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Min<double>(image1, image2, ref outImage);

        return outImage;
    }

    public static CVImagePyramid Min<T>(CVImagePyramid image, T arg1) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Min(image[i], arg1);

        return outImage;
    }

    public static CVImagePyramid Min<T>(CVImagePyramid image, T[] arg1) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Min(image[i], arg1);

        return outImage;
    }

    public static CVImagePyramid Min(CVImagePyramid image1, CVImagePyramid image2)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = Min(image1[i], image2[i]);

        return outImage;
    }
}