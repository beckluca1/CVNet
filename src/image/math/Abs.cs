using System.Numerics;

namespace CVNet;

public static class CVAbs
{
    public static void Abs<T>(
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
                Vector.Abs(vSrc).CopyTo(dst.Slice(i, simdWidth));
            }
        }

        for (; i < count; i++)
        {
            double valueD = (double)Convert.ChangeType(src[i], typeof(double));
            double sqrtD = Math.Abs(valueD);
            T sqrtT = T.CreateChecked(sqrtD);
            dst[i] = sqrtT;
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

    public static CVImagePyramid Abs<T>(CVImagePyramid image) where T : struct, INumber<T>
    {
        CVImagePyramid outImage = new CVImagePyramid(image.Levels);

        for (int i = 0; i < image.Levels; i++)
            outImage[i] = Abs(image[i]);

        return outImage;
    }
}