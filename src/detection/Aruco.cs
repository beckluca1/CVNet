namespace CVNet;

public class CVAruco
{
    public static List<CVContour> GetMarker(CVImage image, int markerSize)
    {
        CVImage gray = CVConvert.ToFormat(image, CVChannelFormat.CV_Grayscale);
        gray = CVProcessing.AdaptiveThresholdMean(gray, 7, 13);

        List<CVContour> contours = CVDetection.FindContours(gray, 1);
        Console.WriteLine($"Contours: {contours.Count}");
        List<CVContour> checkedContours = CVDetection.ContourChecks(contours, image.Width, image.Height);
        Console.WriteLine($"Checked Contours: {checkedContours.Count}");
        List<CVContour> polygons = CVDetection.ApproximatePolygons(checkedContours, 0.01);
        Console.WriteLine($"Polygons: {polygons.Count}");
        List<CVContour> quads = CVDetection.QuadrilateralChecks(polygons, image.Width, image.Height);
        Console.WriteLine($"Quads: {quads.Count}");

        CVDetection.IdentifyContours(image, quads, markerSize);

        return quads;
    }
}