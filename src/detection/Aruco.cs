namespace CVNet;

public class CVAruco
{
    public static List<CVContour> GetMarker(CVImage image, int markerSize)
    {
        CVImage gray = CVConvert.ToFormat(image, CVChannelFormat.CV_Grayscale);
        gray = CVProcessing.AdaptiveThresholdMean(gray, 7, 13);

        List<CVContour> contours = CVDetection.FindContours(gray, 1);
        List<CVContour> polygons = CVDetection.ApproximatePolygons(contours, 0.01);
        List<CVContour> quads = CVDetection.QuadrilateralChecks(polygons, image.Width, image.Height);

        // CVDetection.IdentifyContours(image, quads, markerSize);

        return quads;
    }
}