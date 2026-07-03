#if DEBUG
namespace CVNet;

using Xunit;

public class ArucoTests
{
    [Fact]
    public void TestGetMarker()
    {
        // Create white image and fill with marker like pixels
        CVImage image = CVImage.Create(256, 256, CVColorFormat.CV_C1, CVDataFormat.CV_U8, CVChannelFormat.CV_R, 255);
        Span<byte> imageBuffer = image.BufferAs<byte>();
        for (int y = 30; y < 120; y++)
            for (int x = 30; x < 120; x++)
                imageBuffer[x + y * image.Width] = 0;

        CVAruco.GetMarker(image, 6);
    }
}
#endif