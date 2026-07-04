#if DEBUG
namespace CVNet;

using Xunit;

public class ArucoTests
{
    [Fact]
    public void TestGetMarker()
    {
        // Create white image and fill with marker like pixels
        CVImage image = CVImage.Create<byte>(64, 64, CVColorFormat.CV_C1, CVDataFormat.CV_U8, CVChannelFormat.CV_R, 180);

        Span<byte> imageBuffer = image.BufferAs<byte>();

        int markerSize = 8;
        int boundary = 2;
        int xOffset = 15;
        int yOffset = 24;
        int xSize = 4;
        int ySize = 3;

        byte[] markerPattern = [180, 180, 180, 180, 180, 180, 180, 180,
                                180, 180, 255, 255, 180, 180, 180, 180,
                                180, 180, 255, 180, 255, 180, 255, 180,
                                180, 255, 180, 180, 255, 180, 180, 180,
                                180, 180, 180, 180, 255, 255, 180, 180,
                                180, 255, 180, 180, 255, 255, 180, 180,
                                180, 255, 255, 255, 180, 255, 180, 180,
                                180, 180, 180, 180, 180, 180, 180, 180 ];

        for (int i = 0; i < markerSize * xSize + 2 * boundary; i++)
            for (int j = 0; j < markerSize * ySize + 2 * boundary; j++)
                imageBuffer[(xOffset - boundary + i) + (yOffset - boundary + j) * image.Width] = 255;

        for (int i = 0; i < markerSize; i++)
            for (int j = 0; j < markerSize; j++)
                for (int y = 0; y < ySize; y++)
                    for (int x = 0; x < xSize; x++)
                        imageBuffer[(xOffset + i * xSize + x) + (yOffset + j * ySize + y) * image.Width] = markerPattern[j * markerSize + i];

        CVAruco.GetMarker(image, 6);
    }
}
#endif