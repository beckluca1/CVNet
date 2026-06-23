using System.Numerics;

namespace CVNet;

public class CVMath
{
    public static void Operation<T1, T2>(CVImage imageIn, T2 arg1, Func<T1, T1, T1> operation, ref CVImage imageOut) where T1 : struct where T2 : struct
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<T1> bufferSpanIn = imageIn.BufferAs<T1>();
        Span<T1> bufferSpanOut = imageOut.BufferAs<T1>();

        T1 arg1Conv = (T1)Convert.ChangeType(arg1, typeof(T1));

        for (int i = 0; i < imageDataCount; i++) bufferSpanOut[i] = operation(bufferSpanIn[i], arg1Conv);
    }

    // Optimized
    public static void Operation<T1, T2>(CVImage imageIn, T2[] arg1, Func<T1, T1, T1> operation, ref CVImage imageOut) where T1 : struct where T2 : struct
    {
        int imageDataCount = imageIn.Width * imageIn.Height;

        Span<T1> bufferSpanIn = imageIn.BufferAs<T1>();
        Span<T1> bufferSpanOut = imageOut.BufferAs<T1>();

        for (int c = 0; c < imageIn.Channels; c++)
        {
            int baseIdx = c * imageDataCount;

            T1 arg1Conv = (T1)Convert.ChangeType(arg1[c], typeof(T1));

            for (int i = baseIdx; i < baseIdx + imageDataCount; i++)
            {
                bufferSpanOut[i] = operation(bufferSpanIn[i], arg1Conv);
            }
        }
    }

    public static void Operation<T1, T2>(CVImage imageIn1, CVImage imageIn2, Func<T1, T1, T1> operation, ref CVImage imageOut) where T1 : struct where T2 : struct
    {
        int imageDataCount = imageIn1.Width * imageIn1.Height * imageIn1.Channels;

        Span<T1> bufferSpanIn1 = imageIn1.BufferAs<T1>();
        Span<T2> bufferSpanIn2 = imageIn2.BufferAs<T2>();
        Span<T1> bufferSpanOut = imageOut.BufferAs<T1>();

        for (int i = 0; i < imageDataCount; i++)
            bufferSpanOut[i] = operation(bufferSpanIn1[i], (T1)Convert.ChangeType(bufferSpanIn2[i], typeof(T1)));
    }

    public static void OperationUnsafe<T1>(CVImage imageIn1, CVImage imageIn2, Func<T1, T1, T1> operation, ref CVImage imageOut) where T1 : struct
    {
        int imageDataCount = imageIn1.Width * imageIn1.Height * imageIn1.Channels;

        Span<T1> bufferSpanIn1 = imageIn1.BufferAs<T1>();
        Span<T1> bufferSpanIn2 = imageIn2.BufferAs<T1>();
        Span<T1> bufferSpanOut = imageOut.BufferAs<T1>();

        for (int i = 0; i < imageDataCount; i++)
            bufferSpanOut[i] = operation(bufferSpanIn1[i], bufferSpanIn2[i]);
    }

    public static void Operation<T>(CVImage image1, CVImage image2, Func<T, T, T> operation, ref CVImage outImage) where T : struct
    {
        if (image2.DataFormat == CVDataFormat.CV_U8) Operation<T, byte>(image1, image2, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_S8) Operation<T, sbyte>(image1, image2, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_U16) Operation<T, ushort>(image1, image2, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_S16) Operation<T, short>(image1, image2, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_U32) Operation<T, uint>(image1, image2, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_S32) Operation<T, int>(image1, image2, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_U64) Operation<T, ulong>(image1, image2, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_S64) Operation<T, long>(image1, image2, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_F32) Operation<T, float>(image1, image2, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_F64) Operation<T, double>(image1, image2, operation, ref outImage);
    }

    public static CVImage Add<T>(CVImage image, T arg1) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, AddFunc, ref outImage);

        return outImage;
    }

    public static CVImage Add<T>(CVImage image, T[] arg1) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, AddFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, AddFunc, ref outImage);

        return outImage;
    }

    public static CVImage Add(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) Operation<byte>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Operation<sbyte>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Operation<ushort>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Operation<short>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Operation<uint>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Operation<int>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Operation<ulong>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Operation<long>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Operation<float>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Operation<double>(image1, image2, AddFunc, ref outImage);

        return outImage;
    }

    public static CVImage AddUnsafe(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) OperationUnsafe<byte>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) OperationUnsafe<sbyte>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) OperationUnsafe<ushort>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) OperationUnsafe<short>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) OperationUnsafe<uint>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) OperationUnsafe<int>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) OperationUnsafe<ulong>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) OperationUnsafe<long>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) OperationUnsafe<float>(image1, image2, AddFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) OperationUnsafe<double>(image1, image2, AddFunc, ref outImage);

        return outImage;
    }

    public static CVImage Subtract<T>(CVImage image, T arg1) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, SubtractFunc, ref outImage);

        return outImage;
    }

    public static CVImage Subtract<T>(CVImage image, T[] arg1) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, SubtractFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, SubtractFunc, ref outImage);

        return outImage;
    }

    public static CVImage Subtract(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) Operation<byte>(image1, image2, SubtractFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Operation<sbyte>(image1, image2, SubtractFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Operation<ushort>(image1, image2, SubtractFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Operation<short>(image1, image2, SubtractFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Operation<uint>(image1, image2, SubtractFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Operation<int>(image1, image2, SubtractFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Operation<ulong>(image1, image2, SubtractFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Operation<long>(image1, image2, SubtractFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Operation<float>(image1, image2, SubtractFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Operation<double>(image1, image2, SubtractFunc, ref outImage);

        return outImage;
    }

    public static CVImage Subtract<T>(T arg1, CVImage image) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, SubtractFuncRev, ref outImage);

        return outImage;
    }

    public static CVImage Subtract<T>(T[] arg1, CVImage image) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, SubtractFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, SubtractFuncRev, ref outImage);

        return outImage;
    }

    public static CVImage Multiply<T>(CVImage image, T arg1) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, MultiplyFunc, ref outImage);

        return outImage;
    }

    public static CVImage Multiply<T>(CVImage image, T[] arg1) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, MultiplyFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, MultiplyFunc, ref outImage);

        return outImage;
    }

    public static CVImage Multiply(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) Operation<byte>(image1, image2, MultiplyFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Operation<sbyte>(image1, image2, MultiplyFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Operation<ushort>(image1, image2, MultiplyFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Operation<short>(image1, image2, MultiplyFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Operation<uint>(image1, image2, MultiplyFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Operation<int>(image1, image2, MultiplyFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Operation<ulong>(image1, image2, MultiplyFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Operation<long>(image1, image2, MultiplyFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Operation<float>(image1, image2, MultiplyFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Operation<double>(image1, image2, MultiplyFunc, ref outImage);

        return outImage;
    }

    public static CVImage Divide<T>(CVImage image, T arg1) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, DivideFunc, ref outImage);

        return outImage;
    }

    public static CVImage Divide<T>(CVImage image, T[] arg1) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, DivideFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, DivideFunc, ref outImage);

        return outImage;
    }

    public static CVImage Divide(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) Operation<byte>(image1, image2, DivideFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Operation<sbyte>(image1, image2, DivideFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Operation<ushort>(image1, image2, DivideFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Operation<short>(image1, image2, DivideFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Operation<uint>(image1, image2, DivideFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Operation<int>(image1, image2, DivideFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Operation<ulong>(image1, image2, DivideFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Operation<long>(image1, image2, DivideFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Operation<float>(image1, image2, DivideFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Operation<double>(image1, image2, DivideFunc, ref outImage);

        return outImage;
    }

    public static CVImage Divide<T>(T arg1, CVImage image) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, DivideFuncRev, ref outImage);

        return outImage;
    }

    public static CVImage Divide<T>(T[] arg1, CVImage image) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, DivideFuncRev, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, DivideFuncRev, ref outImage);

        return outImage;
    }

    public static CVImage Max<T>(CVImage image, T arg1) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, MaxFunc, ref outImage);

        return outImage;
    }

    public static CVImage Max<T>(CVImage image, T[] arg1) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, MaxFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, MaxFunc, ref outImage);

        return outImage;
    }

    public static CVImage Max(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) Operation<byte>(image1, image2, MaxFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Operation<sbyte>(image1, image2, MaxFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Operation<ushort>(image1, image2, MaxFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Operation<short>(image1, image2, MaxFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Operation<uint>(image1, image2, MaxFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Operation<int>(image1, image2, MaxFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Operation<ulong>(image1, image2, MaxFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Operation<long>(image1, image2, MaxFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Operation<float>(image1, image2, MaxFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Operation<double>(image1, image2, MaxFunc, ref outImage);

        return outImage;
    }

    public static CVImage Min<T>(CVImage image, T arg1) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, MinFunc, ref outImage);

        return outImage;
    }

    public static CVImage Min<T>(CVImage image, T[] arg1) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, MinFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, MinFunc, ref outImage);

        return outImage;
    }

    public static CVImage Min(CVImage image1, CVImage image2)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) Operation<byte>(image1, image2, MinFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Operation<sbyte>(image1, image2, MinFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Operation<ushort>(image1, image2, MinFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Operation<short>(image1, image2, MinFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Operation<uint>(image1, image2, MinFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Operation<int>(image1, image2, MinFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Operation<ulong>(image1, image2, MinFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Operation<long>(image1, image2, MinFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Operation<float>(image1, image2, MinFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Operation<double>(image1, image2, MinFunc, ref outImage);

        return outImage;
    }
    static T AddFunc<T>(T a, T b) where T : IAdditionOperators<T, T, T> { return a + b; }
    static T SubtractFunc<T>(T a, T b) where T : ISubtractionOperators<T, T, T> { return a - b; }
    static T SubtractFuncRev<T>(T a, T b) where T : ISubtractionOperators<T, T, T> { return b - a; }
    static T MultiplyFunc<T>(T a, T b) where T : IMultiplyOperators<T, T, T> { return a * b; }
    static T DivideFunc<T>(T a, T b) where T : IDivisionOperators<T, T, T> { return a / b; }
    static T DivideFuncRev<T>(T a, T b) where T : IDivisionOperators<T, T, T> { return b / a; }
    static T MaxFunc<T>(T a, T b) where T : IAdditionOperators<T, T, T>, IComparisonOperators<T, T, bool> { return a > b ? a : b; }
    static T MinFunc<T>(T a, T b) where T : IAdditionOperators<T, T, T>, IComparisonOperators<T, T, bool> { return a < b ? a : b; }

    public static void Operation<T1, T2>(CVImage imageIn, T2 arg1, T2 arg2, T2 arg3, Func<T1, T1, T1, T1, T1> operation, ref CVImage imageOut) where T1 : struct where T2 : struct
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<T1> bufferSpanIn = imageIn.BufferAs<T1>();
        Span<T1> bufferSpanOut = imageOut.BufferAs<T1>();

        T1 arg1Conv = (T1)Convert.ChangeType(arg1, typeof(T1));
        T1 arg2Conv = (T1)Convert.ChangeType(arg2, typeof(T1));
        T1 arg3Conv = (T1)Convert.ChangeType(arg3, typeof(T1));

        for (int i = 0; i < imageDataCount; i++) bufferSpanOut[i] = operation(bufferSpanIn[i], arg1Conv, arg2Conv, arg3Conv);
    }

    // Optimized
    public static void Operation<T1, T2>(CVImage imageIn, T2[] arg1, T2[] arg2, T2[] arg3, Func<T1, T1, T1, T1, T1> operation, ref CVImage imageOut) where T1 : struct where T2 : struct
    {
        int imageDataCount = imageIn.Width * imageIn.Height;

        Span<T1> bufferSpanIn = imageIn.BufferAs<T1>();
        Span<T1> bufferSpanOut = imageOut.BufferAs<T1>();

        for (int c = 0; c < imageIn.Channels; c++)
        {
            int baseIdx = c * imageDataCount;

            T1 arg1Conv = (T1)Convert.ChangeType(arg1[c], typeof(T1));
            T1 arg2Conv = (T1)Convert.ChangeType(arg1[c], typeof(T1));
            T1 arg3Conv = (T1)Convert.ChangeType(arg1[c], typeof(T1));

            for (int i = baseIdx; i < baseIdx + imageDataCount; i++)
            {
                bufferSpanOut[i] = operation(bufferSpanIn[i], arg1Conv, arg2Conv, arg3Conv);
            }
        }
    }

    public static void Operation<T1, T2, T3, T4>(CVImage imageIn1, CVImage imageIn2, CVImage imageIn3, CVImage imageIn4, Func<T1, T1, T1, T1, T1> operation, ref CVImage imageOut) where T1 : struct where T2 : struct where T3 : struct where T4 : struct
    {
        int imageDataCount = imageIn1.Width * imageIn1.Height * imageIn1.Channels;

        Span<T1> bufferSpanIn1 = imageIn1.BufferAs<T1>();
        Span<T2> bufferSpanIn2 = imageIn2.BufferAs<T2>();
        Span<T3> bufferSpanIn3 = imageIn3.BufferAs<T3>();
        Span<T4> bufferSpanIn4 = imageIn4.BufferAs<T4>();
        Span<T1> bufferSpanOut = imageOut.BufferAs<T1>();

        for (int i = 0; i < imageDataCount; i++)
            bufferSpanOut[i] = operation(bufferSpanIn1[i],
                                            (T1)Convert.ChangeType(bufferSpanIn2[i], typeof(T1)),
                                            (T1)Convert.ChangeType(bufferSpanIn3[i], typeof(T1)),
                                            (T1)Convert.ChangeType(bufferSpanIn4[i], typeof(T1)));
    }

    public static void Operation<T1, T2, T3>(CVImage imageIn1, CVImage imageIn2, T3 arg2, T3 arg3, Func<T1, T1, T1, T1, T1> operation, ref CVImage imageOut) where T1 : struct where T2 : struct where T3 : struct
    {
        int imageDataCount = imageIn1.Width * imageIn1.Height * imageIn1.Channels;

        Span<T1> bufferSpanIn1 = imageIn1.BufferAs<T1>();
        Span<T2> bufferSpanIn2 = imageIn2.BufferAs<T2>();
        Span<T1> bufferSpanOut = imageOut.BufferAs<T1>();

        T1 arg2Conv = (T1)Convert.ChangeType(arg2, typeof(T1));
        T1 arg3Conv = (T1)Convert.ChangeType(arg3, typeof(T1));

        for (int i = 0; i < imageDataCount; i++)
            bufferSpanOut[i] = operation(bufferSpanIn1[i], (T1)Convert.ChangeType(bufferSpanIn2[i], typeof(T1)), arg2Conv, arg3Conv);

    }

    public static void Operation<T1, T2, T3>(CVImage image1, CVImage image2, CVImage image3, CVImage image4, Func<T1, T1, T1, T1, T1> operation, ref CVImage outImage) where T1 : struct where T2 : struct where T3 : struct
    {
        if (image4.DataFormat == CVDataFormat.CV_U8) Operation<T1, T2, T3, byte>(image1, image2, image3, image4, operation, ref outImage);
        else if (image4.DataFormat == CVDataFormat.CV_S8) Operation<T1, T2, T3, sbyte>(image1, image2, image3, image4, operation, ref outImage);
        else if (image4.DataFormat == CVDataFormat.CV_U16) Operation<T1, T2, T3, ushort>(image1, image2, image3, image4, operation, ref outImage);
        else if (image4.DataFormat == CVDataFormat.CV_S16) Operation<T1, T2, T3, short>(image1, image2, image3, image4, operation, ref outImage);
        else if (image4.DataFormat == CVDataFormat.CV_U32) Operation<T1, T2, T3, uint>(image1, image2, image3, image4, operation, ref outImage);
        else if (image4.DataFormat == CVDataFormat.CV_S32) Operation<T1, T2, T3, int>(image1, image2, image3, image4, operation, ref outImage);
        else if (image4.DataFormat == CVDataFormat.CV_U64) Operation<T1, T2, T3, ulong>(image1, image2, image3, image4, operation, ref outImage);
        else if (image4.DataFormat == CVDataFormat.CV_S64) Operation<T1, T2, T3, long>(image1, image2, image3, image4, operation, ref outImage);
        else if (image4.DataFormat == CVDataFormat.CV_F32) Operation<T1, T2, T3, float>(image1, image2, image3, image4, operation, ref outImage);
        else if (image4.DataFormat == CVDataFormat.CV_F64) Operation<T1, T2, T3, double>(image1, image2, image3, image4, operation, ref outImage);
    }

    public static void Operation<T1, T2>(CVImage image1, CVImage image2, CVImage image3, CVImage image4, Func<T1, T1, T1, T1, T1> operation, ref CVImage outImage) where T1 : struct where T2 : struct
    {
        if (image3.DataFormat == CVDataFormat.CV_U8) Operation<T1, T2, byte>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_S8) Operation<T1, T2, sbyte>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_U16) Operation<T1, T2, ushort>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_S16) Operation<T1, T2, short>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_U32) Operation<T1, T2, uint>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_S32) Operation<T1, T2, int>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_U64) Operation<T1, T2, ulong>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_S64) Operation<T1, T2, long>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_F32) Operation<T1, T2, float>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_F64) Operation<T1, T2, double>(image1, image2, image3, image4, operation, ref outImage);
    }

    public static void Operation<T1>(CVImage image1, CVImage image2, CVImage image3, CVImage image4, Func<T1, T1, T1, T1, T1> operation, ref CVImage outImage) where T1 : struct
    {
        if (image3.DataFormat == CVDataFormat.CV_U8) Operation<T1, byte>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_S8) Operation<T1, sbyte>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_U16) Operation<T1, ushort>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_S16) Operation<T1, short>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_U32) Operation<T1, uint>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_S32) Operation<T1, int>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_U64) Operation<T1, ulong>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_S64) Operation<T1, long>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_F32) Operation<T1, float>(image1, image2, image3, image4, operation, ref outImage);
        else if (image3.DataFormat == CVDataFormat.CV_F64) Operation<T1, double>(image1, image2, image3, image4, operation, ref outImage);
    }

    public static void Operation<T1, T3>(CVImage image1, CVImage image2, T3 arg2, T3 arg3, Func<T1, T1, T1, T1, T1> operation, ref CVImage outImage) where T1 : struct where T3 : struct
    {
        if (image2.DataFormat == CVDataFormat.CV_U8) Operation<T1, byte, T3>(image1, image2, arg2, arg3, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_S8) Operation<T1, sbyte, T3>(image1, image2, arg2, arg3, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_U16) Operation<T1, ushort, T3>(image1, image2, arg2, arg3, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_S16) Operation<T1, short, T3>(image1, image2, arg2, arg3, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_U32) Operation<T1, uint, T3>(image1, image2, arg2, arg3, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_S32) Operation<T1, int, T3>(image1, image2, arg2, arg3, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_U64) Operation<T1, ulong, T3>(image1, image2, arg2, arg3, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_S64) Operation<T1, long, T3>(image1, image2, arg2, arg3, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_F32) Operation<T1, float, T3>(image1, image2, arg2, arg3, operation, ref outImage);
        else if (image2.DataFormat == CVDataFormat.CV_F64) Operation<T1, double, T3>(image1, image2, arg2, arg3, operation, ref outImage);
    }

    public static CVImage Bigger<T>(CVImage image, T arg1, T arg2, T arg3) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);

        return outImage;
    }

    public static CVImage Bigger<T>(CVImage image, T[] arg1, T[] arg2, T[] arg3) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, arg2, arg3, BiggerFunc, ref outImage);

        return outImage;
    }

    public static CVImage Bigger(CVImage image1, CVImage image2, CVImage arg2, CVImage arg3)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) Operation<byte>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Operation<sbyte>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Operation<ushort>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Operation<short>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Operation<uint>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Operation<int>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Operation<ulong>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Operation<long>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Operation<float>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Operation<double>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);

        return outImage;
    }

    public static CVImage Bigger<T>(CVImage image1, CVImage image2, T arg2, T arg3) where T : struct
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image1, image2, arg2, arg3, BiggerFunc, ref outImage);

        return outImage;
    }

    public static CVImage Smaller<T>(CVImage image, T arg1, T arg2, T arg3) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);

        return outImage;
    }

    public static CVImage Smaller<T>(CVImage image, T[] arg1, T[] arg2, T[] arg3) where T : struct
    {
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image, arg1, arg2, arg3, SmallerFunc, ref outImage);

        return outImage;
    }

    public static CVImage Smaller(CVImage image1, CVImage image2, CVImage arg2, CVImage arg3)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) Operation<byte>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Operation<sbyte>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Operation<ushort>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Operation<short>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Operation<uint>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Operation<int>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Operation<ulong>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Operation<long>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Operation<float>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Operation<double>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);

        return outImage;
    }

    public static CVImage Smaller<T>(CVImage image1, CVImage image2, T arg2, T arg3) where T : struct
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.ColorFormat, image1.DataFormat);

        if (image1.DataFormat == CVDataFormat.CV_U8) Operation<byte, T>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S8) Operation<sbyte, T>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U16) Operation<ushort, T>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S16) Operation<short, T>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U32) Operation<uint, T>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S32) Operation<int, T>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_U64) Operation<ulong, T>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_S64) Operation<long, T>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F32) Operation<float, T>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);
        else if (image1.DataFormat == CVDataFormat.CV_F64) Operation<double, T>(image1, image2, arg2, arg3, SmallerFunc, ref outImage);

        return outImage;
    }

    static T BiggerFunc<T>(T a, T b, T arg2, T arg3) where T : IComparisonOperators<T, T, bool> { return (a > b ? arg2 : arg3); }
    static T SmallerFunc<T>(T a, T b, T arg2, T arg3) where T : IComparisonOperators<T, T, bool> { return (a < b ? arg2 : arg3); }
}