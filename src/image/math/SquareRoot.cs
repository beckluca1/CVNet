using System.Numerics;

namespace CVNet;

public static class CVSquareRoot
{
    public static void SquareRoot<T>(
        CVImage imageIn,
        ref CVImage imageOut)
        where T : unmanaged, INumber<T>
    {
        Span<T> src = imageIn.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

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
            double valueD = (double)Convert.ChangeType(src[i], typeof(double));
            double sqrtD = Math.Sqrt(valueD);
            T sqrtT = (T)Convert.ChangeType(sqrtD, typeof(T));
            dst[i] = sqrtT;
        }
    }

    public static CVImage SquareRoot(CVImage image)
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat, image.ChannelFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) SquareRoot<byte>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) SquareRoot<sbyte>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) SquareRoot<ushort>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) SquareRoot<short>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) SquareRoot<uint>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) SquareRoot<int>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) SquareRoot<ulong>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) SquareRoot<long>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) SquareRoot<float>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) SquareRoot<double>(image, ref outImage);

        return outImage;
    }
}