using System.Numerics;

namespace CVNet;

public class CVDebug
{
    public static void PrintImageInfo(CVImage image)
    {
        Console.Write("[Image Info] ");
        Console.Write($"Width: {image.Width} ");
        Console.Write($"Height: {image.Height} ");
        Console.Write($"Channels: {image.Channels} ");
        Console.Write($"Bytes: {image.Bytes} ");
        Console.Write($"DataFormat: {image.DataFormat} ");
        Console.Write($"ColorFormat: {image.ColorFormat} ");
        Console.Write($"ChannelFormat: {image.ChannelFormat} ");
        Console.WriteLine();
    }

    public static void printImageInfoExtended<T>(CVImage image) where T : struct, INumber<T>
    {
        PrintImageInfo(image);

        T min = CVProcessing.MinValue<T>(image);
        T max = CVProcessing.MaxValue<T>(image);

        Console.Write("[Extended Image Info] ");
        Console.Write($"Min Value: {min} ");
        Console.Write($"Max Value: {max} ");
        Console.WriteLine();
    }

    public static void PrintImageInfoExtended(CVImage image)
    {
        if (image.DataFormat == CVDataFormat.CV_U8) printImageInfoExtended<byte>(image);
        else if (image.DataFormat == CVDataFormat.CV_S8) printImageInfoExtended<sbyte>(image);
        else if (image.DataFormat == CVDataFormat.CV_U16) printImageInfoExtended<ushort>(image);
        else if (image.DataFormat == CVDataFormat.CV_S16) printImageInfoExtended<short>(image);
        else if (image.DataFormat == CVDataFormat.CV_U32) printImageInfoExtended<uint>(image);
        else if (image.DataFormat == CVDataFormat.CV_S32) printImageInfoExtended<int>(image);
        else if (image.DataFormat == CVDataFormat.CV_U64) printImageInfoExtended<ulong>(image);
        else if (image.DataFormat == CVDataFormat.CV_S64) printImageInfoExtended<long>(image);
        else if (image.DataFormat == CVDataFormat.CV_F32) printImageInfoExtended<float>(image);
        else if (image.DataFormat == CVDataFormat.CV_F64) printImageInfoExtended<double>(image);
    }

    private static void printImage<T>(CVImage image) where T : struct, INumber<T>
    {
        Span<T> buffer = image.BufferAs<T>();

        for (int y = 0; y < image.Width; y++)
        {
            for (int x = 0; x < image.Height; x++)
            {
                Console.Write(buffer[x + y * image.Width] == T.Zero ? "." : "#");
            }
            Console.WriteLine();
        }
    }

    public static void PrintImage(CVImage image)
    {
        if (image.DataFormat == CVDataFormat.CV_U8) printImage<byte>(image);
        else if (image.DataFormat == CVDataFormat.CV_S8) printImage<sbyte>(image);
        else if (image.DataFormat == CVDataFormat.CV_U16) printImage<ushort>(image);
        else if (image.DataFormat == CVDataFormat.CV_S16) printImage<short>(image);
        else if (image.DataFormat == CVDataFormat.CV_U32) printImage<uint>(image);
        else if (image.DataFormat == CVDataFormat.CV_S32) printImage<int>(image);
        else if (image.DataFormat == CVDataFormat.CV_U64) printImage<ulong>(image);
        else if (image.DataFormat == CVDataFormat.CV_S64) printImage<long>(image);
        else if (image.DataFormat == CVDataFormat.CV_F32) printImage<float>(image);
        else if (image.DataFormat == CVDataFormat.CV_F64) printImage<double>(image);
    }
}