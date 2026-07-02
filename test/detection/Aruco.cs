#if DEBUG
namespace CVNet;

using Xunit;

public class ArucoTests
{
    [Fact]
    public void TestGetMarker()
    {
        CVImage image = CVImage.Create(256, 256, CVColorFormat.CV_C1, CVDataFormat.CV_U8, CVChannelFormat.CV_R);

        CVAruco.GetMarker(image, 6);
    }
}
#endif