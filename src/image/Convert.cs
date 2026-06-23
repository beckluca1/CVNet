using System.Runtime.InteropServices;

namespace CVNet;

public class CVConvert
{
    public static void ConvertBuffer<InT, OutT>(CVImage imageIn, CVImage imageOut) where InT : struct where OutT : struct
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<InT> bufferIn = imageIn.BufferAs<InT>();
        Span<OutT> bufferOut = imageOut.BufferAs<OutT>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = (OutT)Convert.ChangeType(bufferIn[i], typeof(OutT));
    }

    public static void ConvertByteToFloat(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<byte> bufferIn = imageIn.BufferAs<byte>();
        Span<float> bufferOut = imageOut.BufferAs<float>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = bufferIn[i];
    }

    public static void ConvertFloatToByte(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<float> bufferIn = imageIn.BufferAs<float>();
        Span<byte> bufferOut = imageOut.BufferAs<byte>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = (byte)bufferIn[i];
    }

    public static void ConvertByteToInt(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<byte> bufferIn = imageIn.BufferAs<byte>();
        Span<int> bufferOut = imageOut.BufferAs<int>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = bufferIn[i];
    }

    public static void ConvertIntToByte(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<int> bufferIn = imageIn.BufferAs<int>();
        Span<byte> bufferOut = imageOut.BufferAs<byte>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = (byte)bufferIn[i];
    }

    public static void ConvertBuffer<InT>(CVImage imageIn, CVImage imageOut) where InT : struct
    {
        if (imageIn.DataFormat == CVDataFormat.CV_U8 && imageOut.DataFormat == CVDataFormat.CV_F32) ConvertByteToFloat(imageIn, ref imageOut);
        else if (imageIn.DataFormat == CVDataFormat.CV_F32 && imageOut.DataFormat == CVDataFormat.CV_U8) ConvertFloatToByte(imageIn, ref imageOut);
        else if (imageIn.DataFormat == CVDataFormat.CV_U8 && imageOut.DataFormat == CVDataFormat.CV_U32) ConvertByteToInt(imageIn, ref imageOut);
        else if (imageIn.DataFormat == CVDataFormat.CV_U32 && imageOut.DataFormat == CVDataFormat.CV_U8) ConvertIntToByte(imageIn, ref imageOut);

        else if (imageOut.DataFormat == CVDataFormat.CV_U8) ConvertBuffer<InT, byte>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_S8) ConvertBuffer<InT, sbyte>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_U16) ConvertBuffer<InT, ushort>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_S16) ConvertBuffer<InT, short>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_U32) ConvertBuffer<InT, uint>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_S32) ConvertBuffer<InT, int>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_U64) ConvertBuffer<InT, ulong>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_S64) ConvertBuffer<InT, long>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_F32) ConvertBuffer<InT, float>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_F64) ConvertBuffer<InT, double>(imageIn, imageOut);
    }

    public static CVImage ConvertData(CVImage image, CVDataFormat dataFormat)
    {
        CVImage imageOut = CVImage.Create(image.Width, image.Height, image.ColorFormat, dataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) ConvertBuffer<byte>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) ConvertBuffer<sbyte>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) ConvertBuffer<ushort>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) ConvertBuffer<short>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) ConvertBuffer<uint>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) ConvertBuffer<int>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) ConvertBuffer<ulong>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) ConvertBuffer<long>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) ConvertBuffer<float>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) ConvertBuffer<double>(image, imageOut);

        return imageOut;
    }

    private static void CopyChannel(CVImage imageIn, CVImage imageOut, int channelIn, int channelOut)
    {
        int byteCount = imageIn.Width * imageIn.Height * imageIn.Bytes;
        int offsetIn = channelIn * imageIn.Width * imageIn.Height * imageIn.Bytes;
        int offsetOut = channelOut * imageOut.Width * imageOut.Height * imageOut.Bytes;

        Buffer.BlockCopy(imageIn.buffer, offsetIn, imageOut.buffer, offsetOut, byteCount);
    }

    private static void FillChannel<OutT, T>(CVImage imageIn, int channel, T value) where OutT : struct where T : struct
    {
        Span<OutT> bufferSpan = imageIn.BufferAs<OutT>();

        OutT valueConv = (OutT)Convert.ChangeType(value, typeof(OutT));

        bufferSpan.Slice(channel * imageIn.Width * imageIn.Height, imageIn.Width * imageIn.Height).Fill(valueConv);
    }

    private static void FillChannel<T>(CVImage image, int channel, T value) where T : struct
    {
        if (image.DataFormat == CVDataFormat.CV_U8) FillChannel<byte, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_S8) FillChannel<sbyte, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_U16) FillChannel<ushort, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_S16) FillChannel<short, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_U32) FillChannel<uint, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_S32) FillChannel<int, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_U64) FillChannel<ulong, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_S64) FillChannel<long, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_F32) FillChannel<float, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_F64) FillChannel<double, T>(image, channel, value);
    }

    public static CVImage ConvertColor(CVImage image, int[] inChannels)
    {
        CVColorFormat format = (CVColorFormat)inChannels.Length;

        CVImage outImage = CVImage.Create(image.Width, image.Height, format, image.DataFormat);

        for (int i = 0; i < inChannels.Length; i++)
        {
            CopyChannel(image, outImage, inChannels[i], i);
        }

        return outImage;
    }

    public static CVImage RGBToBGR(CVImage image)
    {
        return ConvertColor(image, new int[] { 2, 1, 0 });
    }

    public static CVImage BGRToRGB(CVImage image)
    {
        return ConvertColor(image, new int[] { 2, 1, 0 });
    }

    public static CVImage RGBAToBGR(CVImage image)
    {
        return ConvertColor(image, new int[] { 2, 1, 0 });
    }

    public static CVImage BGRAToRGB(CVImage image)
    {
        return ConvertColor(image, new int[] { 2, 1, 0 });
    }

    public static CVImage RGBAToRGB(CVImage image)
    {
        return ConvertColor(image, new int[] { 0, 1, 2 });
    }

    public static CVImage BGRAToBGR(CVImage image)
    {
        return ConvertColor(image, new int[] { 0, 1, 2 });
    }

    public static CVImage RGBToBGRA<T>(CVImage image, T a) where T : struct
    {
        CVImage swappedImage = ConvertColor(image, new int[] { 2, 1, 0, 0 });

        FillChannel(swappedImage, 3, a);

        return swappedImage;
    }

    public static CVImage RGBToABGR<T>(CVImage image, T a) where T : struct
    {
        CVImage swappedImage = ConvertColor(image, new int[] { 0, 2, 1, 0 });

        FillChannel(swappedImage, 0, a);

        return swappedImage;
    }

    public static CVImage BGRToRGBA<T>(CVImage image, T a) where T : struct
    {
        CVImage swappedImage = ConvertColor(image, new int[] { 2, 1, 0, 0 });

        FillChannel(swappedImage, 3, a);

        return swappedImage;
    }

    public static CVImage BGRToARGB<T>(CVImage image, T a) where T : struct
    {
        CVImage swappedImage = ConvertColor(image, new int[] { 0, 2, 1, 0 });

        FillChannel(swappedImage, 0, a);

        return swappedImage;
    }

    public static CVImage RGBToRGBA<T>(CVImage image, T a) where T : struct
    {
        CVImage swappedImage = ConvertColor(image, new int[] { 0, 1, 2, 0 });

        FillChannel(swappedImage, 3, a);

        return swappedImage;
    }

    public static CVImage RGBToARGB<T>(CVImage image, T a) where T : struct
    {
        CVImage swappedImage = ConvertColor(image, new int[] { 0, 0, 1, 2 });

        FillChannel(swappedImage, 0, a);

        return swappedImage;
    }

    public static CVImage BGRToBGRA<T>(CVImage image, T a) where T : struct
    {
        CVImage swappedImage = ConvertColor(image, new int[] { 0, 1, 2, 0 });

        FillChannel(swappedImage, 3, a);

        return swappedImage;
    }

    public static CVImage BGRToABGR<T>(CVImage image, T a) where T : struct
    {
        CVImage swappedImage = ConvertColor(image, new int[] { 0, 0, 1, 2 });

        FillChannel(swappedImage, 0, a);

        return swappedImage;
    }

    public static CVImage BGRAToGrayscale(CVImage image)
    {
        CVImage R = ConvertColor(image, new int[] { 2 });
        CVImage G = ConvertColor(image, new int[] { 1 });
        CVImage B = ConvertColor(image, new int[] { 0 });

        R = CVDivide.Divide(R, 3);
        G = CVDivide.Divide(G, 3);
        B = CVDivide.Divide(B, 3);

        CVImage RGB = CVAdd.Add(CVAdd.Add(R, G), B);

        return RGB;
    }

    public static CVImage BGRToGrayscale(CVImage image)
    {
        CVImage R = ConvertColor(image, new int[] { 2 });
        CVImage G = ConvertColor(image, new int[] { 1 });
        CVImage B = ConvertColor(image, new int[] { 0 });

        CVImage RGB = CVDivide.Divide(CVAdd.Add(CVAdd.Add(R, G), B), 3);

        return RGB;
    }

    public static CVImage RGBAToGrayscale(CVImage image)
    {
        CVImage R = ConvertColor(image, new int[] { 0 });
        CVImage G = ConvertColor(image, new int[] { 1 });
        CVImage B = ConvertColor(image, new int[] { 2 });

        CVImage RGB = CVDivide.Divide(CVAdd.Add(CVAdd.Add(R, G), B), 3);

        return RGB;
    }

    public static CVImage RGBToGrayscale(CVImage image)
    {
        CVImage R = ConvertColor(image, new int[] { 0 });
        CVImage G = ConvertColor(image, new int[] { 1 });
        CVImage B = ConvertColor(image, new int[] { 2 });

        CVImage RGB = CVDivide.Divide(CVAdd.Add(CVAdd.Add(R, G), B), 3);

        return RGB;
    }

    public static CVImage GrayscaleToRGBA<T>(CVImage image, T a) where T : struct
    {
        CVImage RGB = ConvertColor(image, new int[] { 0, 0, 0, 0 });

        FillChannel(RGB, 3, a);

        return RGB;
    }

    public static CVImage GrayscaleToRGB(CVImage image)
    {
        CVImage RGB = ConvertColor(image, new int[] { 0, 0, 0 });

        return RGB;
    }

    public static CVImage GrayscaleToBGRA<T>(CVImage image, T a) where T : struct
    {
        CVImage RGB = ConvertColor(image, new int[] { 0, 0, 0, 0 });

        FillChannel(RGB, 3, a);

        return RGB;
    }

    public static CVImage GrayscaleToBGR(CVImage image)
    {
        CVImage RGB = ConvertColor(image, new int[] { 0, 0, 0 });

        return RGB;
    }

}