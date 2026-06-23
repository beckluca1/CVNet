using System.Numerics;

namespace CVNet;

public static class CVMax
{
    public static void Max<T, TV>(
        CVImage imageIn,
        TV value,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct
    {
        T valueC = (T)Convert.ChangeType(value, typeof(T));

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
                Vector.Max(vSrc, vValue).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = T.Max(src[i], valueC);
        }
    }

    public static void Max<T, TV>(
        CVImage imageIn,
        TV[] values,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T> where TV : struct
    {
        T[] valuesC = new T[values.Length];
        for (int i = 0; i < valuesC.Length; i++) valuesC[i] = (T)Convert.ChangeType(values[i], typeof(T));

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

                    Vector.Max(vSrc, vValue).CopyTo(dst.Slice(idx, simdWidth));
                }
            }

            for (; i < planeSize; i++)
            {
                int idx = baseIdx + i;
                dst[idx] = T.Max(src[idx], valuesC[c]);
            }
        }
    }

    public static void Max<T>(
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

                Vector.Max(v1, v2).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            dst[i] = T.Max(src1[i], src2[i]);
        }
    }

    public static CVImage Max<TV>(CVImage image, TV arg1) where TV : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Max<byte, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Max<sbyte, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Max<ushort, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Max<short, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Max<uint, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Max<int, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Max<ulong, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Max<long, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Max<float, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Max<double, TV>(image, arg1, ref outImage);

        return outImage;
    }

    public static CVImage Max<TV>(CVImage image, TV[] arg1) where TV : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Max<byte, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Max<sbyte, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Max<ushort, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Max<short, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Max<uint, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Max<int, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Max<ulong, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Max<long, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Max<float, TV>(image, arg1, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Max<double, TV>(image, arg1, ref outImage);

        return outImage;
    }

    public static CVImage Max(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) Max<byte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Max<sbyte>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Max<ushort>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Max<short>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Max<uint>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Max<int>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Max<ulong>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Max<long>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Max<float>(image1, image2, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Max<double>(image1, image2, ref outImage);

        return outImage;
    }
}