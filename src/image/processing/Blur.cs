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

    public static CVImage GaussianBlur(CVImage image, int n)
    {
        CVImage blurMaskX = CVImage.CreateGaussianMask(n, 1, image.ColorFormat, image.DataFormat, image.ChannelFormat);
        CVImage blurMaskY = CVImage.CreateGaussianMask(1, n, image.ColorFormat, image.DataFormat, image.ChannelFormat);

        CVImage imageOut = CVConvolution.ConvolutionX(image, blurMaskX);
        imageOut = CVConvolution.ConvolutionY(imageOut, blurMaskY);

        return imageOut;
    }
}