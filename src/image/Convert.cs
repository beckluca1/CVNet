using System.Numerics;
using System.Runtime.InteropServices;

namespace CVNet;

public class CVConvert
{
    private static void convertDataFormat<InT, OutT>(CVImage imageIn, CVImage imageOut) where InT : struct where OutT : struct
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<InT> bufferIn = imageIn.BufferAs<InT>();
        Span<OutT> bufferOut = imageOut.BufferAs<OutT>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = (OutT)Convert.ChangeType(bufferIn[i], typeof(OutT));
    }

    private static void ConvertByteToFloat(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<byte> bufferIn = imageIn.BufferAs<byte>();
        Span<float> bufferOut = imageOut.BufferAs<float>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = bufferIn[i];
    }

    private static void ConvertFloatToByte(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<float> bufferIn = imageIn.BufferAs<float>();
        Span<byte> bufferOut = imageOut.BufferAs<byte>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = (byte)bufferIn[i];
    }

    private static void ConvertByteToInt(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<byte> bufferIn = imageIn.BufferAs<byte>();
        Span<int> bufferOut = imageOut.BufferAs<int>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = bufferIn[i];
    }

    private static void ConvertIntToByte(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<int> bufferIn = imageIn.BufferAs<int>();
        Span<byte> bufferOut = imageOut.BufferAs<byte>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = (byte)bufferIn[i];
    }

    private static void ConvertByteToUInt(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<byte> bufferIn = imageIn.BufferAs<byte>();
        Span<uint> bufferOut = imageOut.BufferAs<uint>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = bufferIn[i];
    }

    private static void ConvertUIntToByte(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<uint> bufferIn = imageIn.BufferAs<uint>();
        Span<byte> bufferOut = imageOut.BufferAs<byte>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = (byte)bufferIn[i];
    }

    private static void convertDataFormat<InT>(CVImage imageIn, CVImage imageOut) where InT : struct
    {
        if (imageIn.DataFormat == CVDataFormat.CV_U8 && imageOut.DataFormat == CVDataFormat.CV_F32) ConvertByteToFloat(imageIn, ref imageOut);
        else if (imageIn.DataFormat == CVDataFormat.CV_F32 && imageOut.DataFormat == CVDataFormat.CV_U8) ConvertFloatToByte(imageIn, ref imageOut);
        else if (imageIn.DataFormat == CVDataFormat.CV_U8 && imageOut.DataFormat == CVDataFormat.CV_S32) ConvertByteToInt(imageIn, ref imageOut);
        else if (imageIn.DataFormat == CVDataFormat.CV_S32 && imageOut.DataFormat == CVDataFormat.CV_U8) ConvertIntToByte(imageIn, ref imageOut);
        else if (imageIn.DataFormat == CVDataFormat.CV_U8 && imageOut.DataFormat == CVDataFormat.CV_U32) ConvertByteToUInt(imageIn, ref imageOut);
        else if (imageIn.DataFormat == CVDataFormat.CV_U32 && imageOut.DataFormat == CVDataFormat.CV_U8) ConvertUIntToByte(imageIn, ref imageOut);

        else if (imageOut.DataFormat == CVDataFormat.CV_U8) convertDataFormat<InT, byte>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_S8) convertDataFormat<InT, sbyte>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_U16) convertDataFormat<InT, ushort>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_S16) convertDataFormat<InT, short>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_U32) convertDataFormat<InT, uint>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_S32) convertDataFormat<InT, int>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_U64) convertDataFormat<InT, ulong>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_S64) convertDataFormat<InT, long>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_F32) convertDataFormat<InT, float>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_F64) convertDataFormat<InT, double>(imageIn, imageOut);
    }

    public static CVImage ConvertDataFormat(CVImage image, CVDataFormat dataFormat)
    {
        CVImage imageOut = CVImage.Create(image.Width, image.Height, image.ColorFormat, dataFormat, image.ChannelFormat);

        if (dataFormat == image.DataFormat) imageOut = image;

        if (image.DataFormat == CVDataFormat.CV_U8) convertDataFormat<byte>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) convertDataFormat<sbyte>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) convertDataFormat<ushort>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) convertDataFormat<short>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) convertDataFormat<uint>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) convertDataFormat<int>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) convertDataFormat<ulong>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) convertDataFormat<long>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) convertDataFormat<float>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) convertDataFormat<double>(image, imageOut);

        return imageOut;
    }

    public static CVDataFormat GetDataFormatUnsignedInt(int bytesNeeded)
    {
        if (bytesNeeded <= 1) return CVDataFormat.CV_U8;
        else if (bytesNeeded <= 2) return CVDataFormat.CV_U16;
        else if (bytesNeeded <= 4) return CVDataFormat.CV_U32;
        else return CVDataFormat.CV_U64;
        //else throw new Exception($"No Format with {bytesNeeded} bytes");
    }

    public static CVDataFormat GetDataFormatSignedInt(int bytesNeeded)
    {
        if (bytesNeeded <= 1) return CVDataFormat.CV_S8;
        else if (bytesNeeded <= 2) return CVDataFormat.CV_S16;
        else if (bytesNeeded <= 4) return CVDataFormat.CV_S32;
        else return CVDataFormat.CV_S64;
        //else throw new Exception($"No Format with {bytesNeeded} bytes");
    }

    public static CVDataFormat GetDataFormatFloat(int bytesNeeded)
    {
        if (bytesNeeded <= 4) return CVDataFormat.CV_F32;
        else return CVDataFormat.CV_F64;
        //else throw new Exception($"No Format with {bytesNeeded} bytes");
    }

    public static CVImage ConvertDataFormatBytes(CVImage image, int bytesNeeded)
    {
        CVDataFormat dataFormat = image.DataFormat;

        if (image.DataFormat == CVDataFormat.CV_U8) dataFormat = GetDataFormatUnsignedInt(bytesNeeded);
        else if (image.DataFormat == CVDataFormat.CV_S8) dataFormat = GetDataFormatSignedInt(bytesNeeded);
        else if (image.DataFormat == CVDataFormat.CV_U16) dataFormat = GetDataFormatUnsignedInt(bytesNeeded);
        else if (image.DataFormat == CVDataFormat.CV_S16) dataFormat = GetDataFormatSignedInt(bytesNeeded);
        else if (image.DataFormat == CVDataFormat.CV_U32) dataFormat = GetDataFormatUnsignedInt(bytesNeeded);
        else if (image.DataFormat == CVDataFormat.CV_S32) dataFormat = GetDataFormatSignedInt(bytesNeeded);
        else if (image.DataFormat == CVDataFormat.CV_U64) dataFormat = GetDataFormatUnsignedInt(bytesNeeded);
        else if (image.DataFormat == CVDataFormat.CV_S64) dataFormat = GetDataFormatSignedInt(bytesNeeded);
        else if (image.DataFormat == CVDataFormat.CV_F32) dataFormat = GetDataFormatFloat(bytesNeeded);
        else if (image.DataFormat == CVDataFormat.CV_F64) dataFormat = GetDataFormatFloat(bytesNeeded);

        return ConvertDataFormat(image, dataFormat);
    }

    private static int getBytesRequiredUnsigned<T>(CVImage image) where T : struct, INumber<T>
    {
        double min = (double)Convert.ChangeType(CVProcessing.MinValue<T>(image), typeof(double));
        double max = (double)Convert.ChangeType(CVProcessing.MaxValue<T>(image), typeof(double));

        double bytes = Math.Log2(max) / 8;

        return (int)Math.Ceiling(bytes);

    }

    private static int getBytesRequiredSigned<T>(CVImage image) where T : struct, INumber<T>
    {
        double min = (double)Convert.ChangeType(CVProcessing.MinValue<T>(image), typeof(double));
        double max = (double)Convert.ChangeType(CVProcessing.MaxValue<T>(image), typeof(double));

        double bytes = Math.Log2(Math.Max(max, -min)) / 4;

        return (int)Math.Ceiling(bytes);
    }

    public static int GetBytesRequired(CVImage image)
    {
        if (image.DataFormat == CVDataFormat.CV_U8) return getBytesRequiredUnsigned<byte>(image);
        else if (image.DataFormat == CVDataFormat.CV_S8) return getBytesRequiredSigned<sbyte>(image);
        else if (image.DataFormat == CVDataFormat.CV_U16) return getBytesRequiredUnsigned<ushort>(image);
        else if (image.DataFormat == CVDataFormat.CV_S16) return getBytesRequiredSigned<short>(image);
        else if (image.DataFormat == CVDataFormat.CV_U32) return getBytesRequiredUnsigned<uint>(image);
        else if (image.DataFormat == CVDataFormat.CV_S32) return getBytesRequiredSigned<int>(image);
        else if (image.DataFormat == CVDataFormat.CV_U64) return getBytesRequiredUnsigned<ulong>(image);
        else if (image.DataFormat == CVDataFormat.CV_S64) return getBytesRequiredSigned<long>(image);
        // Floating point values dont "require" bits
        else if (image.DataFormat == CVDataFormat.CV_F32) return 8;
        else if (image.DataFormat == CVDataFormat.CV_F64) return 16;

        return 0;
    }

    public static int GetDataFormatFactorBytes<T>(CVImage image, double factor) where T : struct, INumber<T>
    {
        int requiredBytes = GetBytesRequired(image);
        double addedBytes = Math.Log2(factor) / 8.0;
        // Log rule: add required and added bytes
        int newBytes = (int)Math.Ceiling(requiredBytes + addedBytes);
        return newBytes;
    }

    public static CVImage ConvertDataFormatFactor(CVImage image, double factor)
    {
        int newBytes = image.Bytes;

        if (image.DataFormat == CVDataFormat.CV_U8) newBytes = GetDataFormatFactorBytes<byte>(image, factor);
        else if (image.DataFormat == CVDataFormat.CV_S8) newBytes = GetDataFormatFactorBytes<sbyte>(image, factor);
        else if (image.DataFormat == CVDataFormat.CV_U16) newBytes = GetDataFormatFactorBytes<ushort>(image, factor);
        else if (image.DataFormat == CVDataFormat.CV_S16) newBytes = GetDataFormatFactorBytes<short>(image, factor);
        else if (image.DataFormat == CVDataFormat.CV_U32) newBytes = GetDataFormatFactorBytes<uint>(image, factor);
        else if (image.DataFormat == CVDataFormat.CV_S32) newBytes = GetDataFormatFactorBytes<int>(image, factor);
        else if (image.DataFormat == CVDataFormat.CV_U64) newBytes = GetDataFormatFactorBytes<ulong>(image, factor);
        else if (image.DataFormat == CVDataFormat.CV_S64) newBytes = GetDataFormatFactorBytes<long>(image, factor);
        else if (image.DataFormat == CVDataFormat.CV_F32) newBytes = GetDataFormatFactorBytes<float>(image, factor);
        else if (image.DataFormat == CVDataFormat.CV_F64) newBytes = GetDataFormatFactorBytes<double>(image, factor);

        return ConvertDataFormatBytes(image, newBytes);
    }

    public static CVImage ConvertDataFormatToSigned(CVImage image)
    {
        CVDataFormat dataFormat = image.DataFormat;

        // Double the bytes so large values are consistent except for U64
        if (image.DataFormat == CVDataFormat.CV_U8) dataFormat = CVDataFormat.CV_S16;
        else if (image.DataFormat == CVDataFormat.CV_U16) dataFormat = CVDataFormat.CV_S32;
        else if (image.DataFormat == CVDataFormat.CV_U32) dataFormat = CVDataFormat.CV_S64;
        else if (image.DataFormat == CVDataFormat.CV_U64) dataFormat = CVDataFormat.CV_S64;

        return ConvertDataFormat(image, dataFormat);
    }

    public static CVImage ConvertDataFormatToFloat(CVImage image)
    {
        CVDataFormat dataFormat = image.DataFormat;

        // Keep the number of bytes where possible
        if (image.DataFormat == CVDataFormat.CV_U8) dataFormat = CVDataFormat.CV_F32;
        else if (image.DataFormat == CVDataFormat.CV_S8) dataFormat = CVDataFormat.CV_F32;
        else if (image.DataFormat == CVDataFormat.CV_U16) dataFormat = CVDataFormat.CV_F32;
        else if (image.DataFormat == CVDataFormat.CV_S16) dataFormat = CVDataFormat.CV_F32;
        else if (image.DataFormat == CVDataFormat.CV_U32) dataFormat = CVDataFormat.CV_F32;
        else if (image.DataFormat == CVDataFormat.CV_S32) dataFormat = CVDataFormat.CV_F32;
        else if (image.DataFormat == CVDataFormat.CV_U64) dataFormat = CVDataFormat.CV_F64;
        else if (image.DataFormat == CVDataFormat.CV_S64) dataFormat = CVDataFormat.CV_F64;

        return ConvertDataFormat(image, dataFormat);
    }


    public static void CopyChannel(CVImage imageIn, CVImage imageOut, int channelIn, int channelOut)
    {
        int byteCount = imageIn.Width * imageIn.Height * imageIn.Bytes;
        int offsetIn = channelIn * imageIn.Width * imageIn.Height * imageIn.Bytes;
        int offsetOut = channelOut * imageOut.Width * imageOut.Height * imageOut.Bytes;

        Buffer.BlockCopy(imageIn.buffer, offsetIn, imageOut.buffer, offsetOut, byteCount);
    }

    private static void fillChannel<OutT, T>(CVImage imageIn, int channel, T value) where OutT : struct where T : struct
    {
        Span<OutT> bufferSpan = imageIn.BufferAs<OutT>();

        OutT valueConv = (OutT)Convert.ChangeType(value, typeof(OutT));

        bufferSpan.Slice(channel * imageIn.Width * imageIn.Height, imageIn.Width * imageIn.Height).Fill(valueConv);
    }

    public static void FillChannel<T>(CVImage image, int channel, T value) where T : struct
    {
        if (image.DataFormat == CVDataFormat.CV_U8) fillChannel<byte, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_S8) fillChannel<sbyte, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_U16) fillChannel<ushort, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_S16) fillChannel<short, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_U32) fillChannel<uint, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_S32) fillChannel<int, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_U64) fillChannel<ulong, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_S64) fillChannel<long, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_F32) fillChannel<float, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_F64) fillChannel<double, T>(image, channel, value);
    }

    public static CVImage ConvertColorFormat(CVImage image, int[] inChannels)
    {
        CVColorFormat format = (CVColorFormat)inChannels.Length;

        CVImage outImage = CVImage.Create(image.Width, image.Height, format, image.DataFormat, image.ChannelFormat);

        for (int i = 0; i < inChannels.Length; i++)
        {
            CopyChannel(image, outImage, inChannels[i], i);
        }

        return outImage;
    }

    public static CVImage AverageChannels(CVImage image, CVChannelFormat channelFormat)
    {
        // Expand so sum doesnt fail
        // Max value can be max three times the input
        CVImage expanded = ConvertDataFormatFactor(image, image.Channels);

        CVChannelFormats channelFormats = new CVChannelFormats(channelFormat);
        Dictionary<CVChannel, CVImage> uniqueChannels = new Dictionary<CVChannel, CVImage>();
        for (int i = 0; i < channelFormats.Channels.Length; i++)
        {
            if (uniqueChannels.ContainsKey(channelFormats.Channels[i])) continue;

            for (int j = 0; j < expanded.ChannelFormats.Channels.Length; j++)
            {
                if (channelFormats.Channels[i] == expanded.ChannelFormats.Channels[j])
                {
                    uniqueChannels.Add(channelFormats.Channels[i], ConvertColorFormat(expanded, [j]));
                    break;
                }
            }
        }

        if (uniqueChannels.Count == 0) throw new Exception("No RGB components found");

        CVImage sum = uniqueChannels.Values.ElementAt(0);
        sum.ChannelFormat = CVChannelFormat.CV_R;
        for (int i = 1; i < uniqueChannels.Count; i++)
        {
            sum = sum + uniqueChannels.Values.ElementAt(i);
        }
        sum = CVDivide.Divide(sum, uniqueChannels.Count);

        //Recast to original type
        sum = ConvertDataFormat(sum, image.DataFormat);

        return sum;
    }

    private static void convertChannelFormat<T>(CVImage image, ref CVImage imageOut) where T : struct, INumber<T>
    {
        int[] requestedChannels = new int[imageOut.ChannelFormats.Channels.Length];
        for (int i = 0; i < imageOut.ChannelFormats.Channels.Length; i++)
            for (int j = 0; j < image.ChannelFormats.Channels.Length; j++)
            {
                if (imageOut.ChannelFormats.Channels[i] == image.ChannelFormats.Channels[j])
                {
                    requestedChannels[i] = j;
                    break;
                }
            }

        imageOut = ConvertColorFormat(image, requestedChannels);

        for (int i = 0; i < imageOut.ChannelFormats.Channels.Length; i++)
        {
            if (imageOut.ChannelFormats.Channels[i] == CVChannel.CV_AVG_RGB)
            {
                CVImage average = AverageChannels(image, CVChannelFormat.CV_RGB);

                CopyChannel(average, imageOut, 0, i);
            }
            else if (imageOut.ChannelFormats.Channels[i] == CVChannel.CV_AVG_RGBA)
            {
                CVImage average = AverageChannels(image, CVChannelFormat.CV_RGBA);
                CopyChannel(average, imageOut, 0, i);
            }
            else if (imageOut.ChannelFormats.Channels[i] == CVChannel.CV_A_ZERO)
                FillChannel(imageOut, i, 0);
            else if (imageOut.ChannelFormats.Channels[i] == CVChannel.CV_A_ONE)
                FillChannel(imageOut, i, 1);
            else if (imageOut.ChannelFormats.Channels[i] == CVChannel.CV_A_255)
                FillChannel(imageOut, i, 255);
        }

        // Replace Placeholder Channel Types
        for (int i = 0; i < imageOut.ChannelFormats.Channels.Length; i++)
        {
            if (imageOut.ChannelFormats.Channels[i] >= CVChannel.CV_AVG_RGB && imageOut.ChannelFormats.Channels[i] <= CVChannel.CV_AVG_RGBA)
                imageOut.ChannelFormats.Channels[i] = CVChannel.CV_R;
            else if (imageOut.ChannelFormats.Channels[i] >= CVChannel.CV_A_ZERO && imageOut.ChannelFormats.Channels[i] <= CVChannel.CV_A_255)
                imageOut.ChannelFormats.Channels[i] = CVChannel.CV_A;
        }
    }

    public static CVImage ConvertChannelFormat(CVImage image, CVChannelFormat channelFormat)
    {
        CVImage imageOut = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat, channelFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) convertChannelFormat<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) convertChannelFormat<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) convertChannelFormat<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) convertChannelFormat<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) convertChannelFormat<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) convertChannelFormat<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) convertChannelFormat<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) convertChannelFormat<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) convertChannelFormat<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) convertChannelFormat<double>(image, ref imageOut);

        return imageOut;
    }
}