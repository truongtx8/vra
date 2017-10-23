function BaiTap19()
    imgI = imread('cameraman.jpg');
    
    arrPointI = detectSURFFeatures(imgI);
    [arrfeaturesI, arrValidPointsI] = extractFeatures(imgI, arrPointI);

    figure;
    imshow(imgI);
    arrSubValidPointI = arrValidPointsI.selectStrongest(20);
    
    hold on;

    plot(arrSubValidPointI);
end