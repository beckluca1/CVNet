using System.Numerics;

namespace CVNet;

public static partial class CVMath
{
    public static void Greater<T, TV>(
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

                var mask = Vector.GreaterThan(vSrc, vValue);

                var ones = Vector<T>.One;
                var zeros = Vector<T>.Zero;

                var result = Vector.ConditionalSelect(mask, ones, zeros);

                result.CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = src[i] > valueC ? T.One : T.Zero;
        }
    }

    public static void Greater<T, TV>(
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

                    var mask = Vector.GreaterThan(vSrc, vValue);

                    var ones = Vector<T>.One;
                    var zeros = Vector<T>.Zero;

                    var result = Vector.ConditionalSelect(mask, ones, zeros);

                    result.CopyTo(dst.Slice(idx, simdWidth));
                }
            }

            for (; i < planeSize; i++)
            {
                int idx = baseIdx + i;
                dst[idx] = src[idx] > valuesC[idx] ? T.One : T.Zero;
            }
        }
    }

    public static void Greater<T>(
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

                var mask = Vector.GreaterThan(v1, v2);

                var ones = Vector<T>.One;
                var zeros = Vector<T>.Zero;

                var result = Vector.ConditionalSelect(mask, ones, zeros);

                result.CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = src1[i] > src2[i] ? T.One : T.Zero;
        }
    }

    public static CVImage Greater<T>(CVImage image, T arg1) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Greater<byte, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Greater<sbyte, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Greater<ushort, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Greater<short, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Greater<uint, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Greater<int, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Greater<ulong, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Greater<long, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Greater<float, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Greater<double, T>(image, arg1, ref outImage);

        return outImage;
    }

    public static CVImage Greater<T>(CVImage image, T[] arg1) where T : struct, INumber<T>
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) Greater<byte, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Greater<sbyte, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Greater<ushort, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Greater<short, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Greater<uint, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Greater<int, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Greater<ulong, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Greater<long, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Greater<float, T>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Greater<double, T>(image, arg1, ref outImage);

        return outImage;
    }

    public static CVImage Greater(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) Greater<byte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Greater<sbyte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Greater<ushort>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Greater<short>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Greater<uint>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Greater<int>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Greater<ulong>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Greater<long>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Greater<float>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Greater<double>(image1, image2, ref outImage);

        return outImage;
    }

    public static CVImagePyramid Greater<T>(CVImagePyramid image, T arg1) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Greater(image[i], arg1);

        return outImage;
    }

    public static CVImagePyramid Greater<T>(CVImagePyramid image, T[] arg1) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Greater(image[i], arg1);

        return outImage;
    }

    public static CVImagePyramid Greater(CVImagePyramid image1, CVImagePyramid image2)
    {
        CVImagePyramid outImage = new CVImagePyramid(image1.Levels);

        for (int i = 0; i < image1.Levels; i++)
            outImage[i] = Greater(image1[i], image2[i]);

        return outImage;
    }
}