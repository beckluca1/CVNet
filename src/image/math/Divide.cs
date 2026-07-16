using System.Numerics;

namespace CVNet;

public static class CVDivide
{
    public static void Divide<T, TV>(
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
                (vSrc / vValue).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = src[i] / valueC;
        }
    }

    public static void Divide<T, TV>(
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

                    (vSrc / vValue)
                        .CopyTo(dst.Slice(idx, simdWidth));
                }
            }

            for (; i < planeSize; i++)
            {
                int idx = baseIdx + i;
                dst[idx] = src[idx] / valuesC[c];
            }
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

        int count = src1.Length;

        int simdWidth = Vector<T>.Count;
        int i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= count - simdWidth; i += simdWidth)
            {
                Vector<T> v1 = new(src1.Slice(i, simdWidth));
                Vector<T> v2 = new(src2.Slice(i, simdWidth));

                (v1 / v2).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = src1[i] / src2[i];
        }
    }

    public static CVImage Divide<TV>(CVImage image, TV arg1) where TV : struct, INumber<TV>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Divide<byte, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Divide<sbyte, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Divide<ushort, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Divide<short, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Divide<uint, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Divide<int, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Divide<ulong, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Divide<long, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Divide<float, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Divide<double, TV>(image, arg1, ref outImage);

        return outImage;
    }

    public static CVImage Divide<TV>(CVImage image, TV[] arg1) where TV : struct, INumber<TV>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Divide<byte, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Divide<sbyte, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Divide<ushort, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Divide<short, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Divide<uint, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Divide<int, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Divide<ulong, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Divide<long, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Divide<float, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Divide<double, TV>(image, arg1, ref outImage);

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

    public static CVImagePyramid Divide<TV>(CVImagePyramid image, TV arg1) where TV : struct, INumber<TV>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Divide(image[i], arg1);

        return outImage;
    }

    public static CVImagePyramid Divide<TV>(CVImagePyramid image, TV[] arg1) where TV : struct, INumber<TV>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Divide(image[i], arg1);

        return outImage;
    }

    public static CVImagePyramid Divide(CVImagePyramid image1, CVImagePyramid image2)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = Divide(image1[i], image2[i]);

        return outImage;
    }

    /*public static CVImagePyramid Divide<TV>(CVImagePyramid image, TV arg1, int channel) where TV : struct, INumber<TV>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Divide(image[i], arg1, channel);

        return outImage;
    }

    public static CVImagePyramid Divide(CVImagePyramid image1, CVImagePyramid image2, int channel)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = Divide(image1[i], image2[i], channel);

        return outImage;
    }*/
}