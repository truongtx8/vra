function BaiTap16()
    imgI = imread('cameraman.tif');
    
    imgI = rgb2gray(imgI);
    imwrite(imgI, 'cameraman.jpg')
	subplot (1, 2, 1);
    imshow (imgI);
    
    imgJ = imrotate(imgI, 30);
    imwrite(imgJ, 'cameraman30.jpg')
    subplot (1, 2, 2);
    imshow (imgJ);

end