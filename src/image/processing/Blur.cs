namespace CVNet;

public class CVBlur
{
    public static CVImage MeanBlur(CVImage image, int n)
    {
        // Sum window casts type so it has space for the sum
        CVImage imageOut = CVProcessing.SumWindow(image, n);
        imageOut = CVDivide.Divide(imageOut, n * n);
        // Recast to original type
        imageOut = CVConvert.ConvertDataFormat(imageOut, image.DataFormat);
        return imageOut;
    }

    public static CVImage GaussianBlur(CVImage image, int n, double sigma = 0.0)
    {
        if (sigma == 0.0)
            sigma = (n - 1) / 6.0;

        CVImage blurMaskX = CVImage.CreateGaussianMask(n, 1, image.DataFormat, image.ChannelFormats, sigma);
        CVImage blurMaskY = CVImage.CreateGaussianMask(1, n, image.DataFormat, image.ChannelFormats, sigma);

        CVImage imageOut = CVConvolution.ConvolutionX(image, blurMaskX);
        imageOut = CVConvolution.ConvolutionY(imageOut, blurMaskY);

        return imageOut;
    }
}