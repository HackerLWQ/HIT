function y=imgprocessing(I)
    I=I(:,:,1);
    [y, ~] = imgradient(I,'sobel');
    y=y*255/(max(max(y)));
    y=round(y);
    y=uint8(y);
end