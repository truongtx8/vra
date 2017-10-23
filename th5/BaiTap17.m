function BaiTap17()
    imgI = imread('cameraman.jpg');
    
    imshow(imgI);
    
    points = detectSURFFeatures(imgI);
    
    hold on;
    subpoints = points.selectStrongest(20);
    
    plot(subpoints);

end