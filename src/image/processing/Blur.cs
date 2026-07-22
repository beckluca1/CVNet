namespace CVNet;

public class CVBlur
{
    public static CVImage MeanBlur(CVImage image, int radius)
    {
        CVImage imageOut = CVWindowing.AverageWindow(image, radius);
        return imageOut;
    }

    public static CVImage GaussianBlur(CVImage image, int radius, double sigma = 0.0)
    {
        if (sigma == 0.0)
            sigma = radius / 3.0;

        CVImage doubleImage = CVConvert.ConvertDataFormat(image, CVDataFormat.CV_F64);

        CVImage blurMaskX = CVImage.CreateGaussianMask(radius * 2 + 1, 1, sigma);
        CVImage blurMaskY = CVImage.CreateGaussianMask(1, radius * 2 + 1, sigma);

        CVImage imageOut = CVConvolution.ConvolutionX(doubleImage, blurMaskX);
        imageOut = CVConvolution.ConvolutionY(imageOut, blurMaskY);

        return CVConvert.ConvertDataFormat(imageOut, image.DataFormat);
    }
}