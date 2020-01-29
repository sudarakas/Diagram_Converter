%Clear the Screen & Terminal
clc;
close all;
clear;

%Input the image & create the output folder
imgName = 'r'; %image name
inputFile = strcat(imgName, '.jpg');
inputFolder = './input';
outputFolder = strcat('./output/', imgName); 
inputLocation = fullfile(inputFolder, inputFile); 
mkdir(outputFolder);

%Display image
inputImage = imread(inputLocation); 

%Convert image to grayscale image
grayImage = rgb2gray(inputImage);

%Reduce the image size
[rows,cols] = size(grayImage);
rows = rows/10; 
cols = cols/10;
resizeImage = imresize(grayImage, [rows, cols]);

%Convert the gray image to binary image
thresholding = adaptthresh(resizeImage, 0.3, 'ForegroundPolarity','dark');
binaryImage = imbinarize(resizeImage, thresholding);
outputFile = 'Binary Image.jpg';
outputLocation = fullfile(outputFolder, outputFile);
saveas(gcf, outputLocation);

%Subplot the figures
subplot(2,2,1);imshow(inputImage);title('Input Image');
subplot(2,2,2);imshow(grayImage);title('Grayscale Image');
subplot(2,2,3);imshow(resizeImage);title('Size Reduced Image');
subplot(2,2,4);imshow(binaryImage);title('Binary Image');

%Bitwise Inversion
invertedImage = 1-binaryImage;
outputFile = 'Bitwise Inverted Image.jpg';
outputLocation = fullfile(outputFolder, outputFile);
saveas(gcf, outputLocation);

%Noise Reduction
connectedComponents = bwconncomp(invertedImage);
stats = regionprops(connectedComponents, 'Area');
labelMatrix = labelmatrix(connectedComponents);
noiseReducedImage = ismember(labelMatrix, find([stats.Area] >= 50*rows/100));
outputFile = 'Noise Reduced Image.jpg';
outputLocation = fullfile(outputFolder, outputFile);
saveas(gcf, outputLocation);

%Fill the image
filledImage = imfill(noiseReducedImage, 'holes');
outputFile = 'Filled Image.jpg';
outputLocation = fullfile(outputFolder, outputFile);
saveas(gcf, outputLocation);

%Detect the edges of the objects
detectedEdges = edge(filledImage, 'zerocross');
outputFile = 'Detected Edges.jpg';
outputLocation = fullfile(outputFolder, outputFile);
saveas(gcf, outputLocation);

%Subplot the figures
figure;
subplot(2,2,1);imshow(invertedImage);title('Bitwise Inverted Image');
subplot(2,2,2);imshow(noiseReducedImage);title('Noise Reduced Image');
subplot(2,2,3);imshow(filledImage);title('Filled Image');
subplot(2,2,4);imshow(detectedEdges);title('Detected Edges');

%Hough Transform Mechanism
[H,theta,rho] = hough(detectedEdges);
topPoints = houghpeaks(H, 100);

detectedlLines = houghlines(detectedEdges,theta,rho,topPoints,'FillGap',5,'MinLength',2); 
figure;subplot(2,2,1); imshow(noiseReducedImage); hold on

for i = 1:length(detectedlLines)
   xy = [detectedlLines(i).point1; detectedlLines(i).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','blue');
   
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','green');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
   
end
title('Lines Detected Image');
outputFile = 'Lines Detected Image.jpg';
outputLocation = fullfile(outputFolder, outputFile);
saveas(gcf, outputLocation);

% Find appropriate angle and rotate lines
detectedlLines = houghlines(detectedEdges, theta, rho, topPoints);

appropriateAngle = mode([detectedlLines.theta])+90;
rotatedImage = imrotate(noiseReducedImage, appropriateAngle);
rotatedFilledImage = imrotate(filledImage, appropriateAngle);
outputFile = 'Rotated Image.jpg';
outputLocation = fullfile(outputFolder, outputFile);
saveas(gcf, outputLocation);

[rows, cols] = size(rotatedImage);

outputFile = 'Rotated Filled Image.jpg';
outputLocation = fullfile(outputFolder, outputFile);
saveas(gcf, outputLocation);

%Seperate Shapes and Arrow Heads
se = strel('diamond', 5);
openedImage = imopen(rotatedFilledImage, se);
removedCCImage = bwareaopen(openedImage, 50);

arrayImage = rotatedImage - removedCCImage;
arrayImage = imbinarize(arrayImage);

arrows = bwareaopen(arrayImage, 20);
shapes = rotatedImage - arrows;

outputFile = 'Detected Shapes.jpg';
outputLocation = fullfile(outputFolder, outputFile);
saveas(gcf, outputLocation);

%Subplot the figures (2,2,1 already plotted)
subplot(2,2,2);imshow(rotatedImage);title('Rotated Image');
subplot(2,2,3);imshow(rotatedFilledImage);title('Rotated Filled Image');
subplot(2,2,4);imshow(shapes);title('Detected Shapes');

%Detect Rectangales, Circles, Diamonds
[shape, noShape] = bwlabel(shapes);
figure;
subplot(1,2,1);imagesc(shape);
axis equal;
title('Unique Shapes');
outputFile = 'Unique Shapes.jpg';
outputLocation = fullfile(outputFolder, outputFile);
saveas(gcf, outputLocation);

shapesProperties  = regionprops(shape, 'all');

shapesCentroids = cat(1, shapesProperties.Centroid);
shapesPerimeters = cat(1, shapesProperties.Perimeter);
shapesArea = cat(1, shapesProperties.ConvexArea);
shapeBBs = cat(1, shapesProperties.BoundingBox);

circlesRatio = (shapesPerimeters.^2)./(4*pi*shapesArea);
rectangleRatio = NaN(noShape,1);

for i = 1:noShape
    [p,q] = size(shapesProperties(i).FilledImage);
    rectangleRatio(i) = shapesArea(i)/(p*q);
end

%Identify the shapes of the image
shapeCircle = (circlesRatio < 1.1);
shapeRectangale = (rectangleRatio > 0.75);
shapeRectangale = logical(shapeRectangale .* ~shapeCircle);
shapeDiamond = (rectangleRatio <= 0.75);
shapeDiamond = logical(shapeDiamond .* ~shapeCircle);


%Identify arrows
[arrowL, arrowN] = bwlabel(arrows);
subplot(1,2,2);imagesc(arrowL);
axis equal;
title('Arrows Heads and Tails');
arrowProperties = regionprops(arrowL, 'all');

arrowCentroids = cat(1, arrowProperties.Centroid);
arrowBoundingBoxes = cat(1, arrowProperties.BoundingBox);
arrowCentres = [arrowBoundingBoxes(:, 1) + 0.5*arrowBoundingBoxes(:, 3), arrowBoundingBoxes(:, 2) + 0.5*arrowBoundingBoxes(:, 4)]; %Centre of Bounding Box of each arrow

arrowBoundingBoxesMidpts = [];
arrowHeadsAll = [];
arrowTailsAll = [];

for i = 1:arrowN
    hold on;
    arrowOrient = arrowProperties(i).Orientation;
    if (abs(abs(arrowOrient)-90) > abs(arrowOrient))
        arrowBoundingBoxMidpt = [arrowBoundingBoxes(i, 1), arrowCentres(i, 2);  arrowBoundingBoxes(i, 1) + arrowBoundingBoxes(i, 3), arrowCentres(i, 2)];
    else 
        arrowBoundingBoxMidpt = [arrowCentres(i, 1), arrowBoundingBoxes(i, 2); arrowCentres(i, 1), arrowBoundingBoxes(i, 2) + arrowBoundingBoxes(i, 4)];
    end
    
    if (pdist([arrowCentroids(i, :); arrowBoundingBoxMidpt(1, :)], 'euclidean') <= pdist([arrowCentres(i, :); arrowBoundingBoxMidpt(1, :)], 'euclidean'))
        arrowHead = arrowBoundingBoxMidpt(1, :);
        arrowTail = arrowBoundingBoxMidpt(2, :);
    else
        arrowHead = arrowBoundingBoxMidpt(2, :);
        arrowTail = arrowBoundingBoxMidpt(1, :);
    end
    
    plot(arrowHead(:, 1), arrowHead(:, 2), 'g*', 'LineWidth', 2, 'MarkerSize', 5);
    plot(arrowTail(:, 1), arrowTail(:, 2), 'y*', 'LineWidth', 2, 'MarkerSize', 5);
    
    arrowBoundingBoxesMidpts = [arrowBoundingBoxesMidpts; arrowBoundingBoxMidpt];
    arrowHeadsAll = [arrowHeadsAll; arrowHead];
    arrowTailsAll = [arrowTailsAll; arrowTail];
end

%Apply Arrow Head
shapeBoundingBoxesMidpts = [];

for i = 1:noShape
    
    shapeBoundingBox = shapesProperties(i).BoundingBox;

    shapeBBMidpt1 = [shapeBoundingBox(1) + 0.5*shapeBoundingBox(3), shapeBoundingBox(2)];
    shapeBBMidpt2 = [shapeBoundingBox(1) + shapeBoundingBox(3), shapeBoundingBox(2) + 0.5*shapeBoundingBox(4)];
    shapeBBMidpt3 = [shapeBoundingBox(1) + 0.5*shapeBoundingBox(3), shapeBoundingBox(2) + shapeBoundingBox(4)];
    shapeBBMidpt4 = [shapeBoundingBox(1), shapeBoundingBox(2) + 0.5*shapeBoundingBox(4)];
    shapeBoundingBoxesMidpts = [shapeBoundingBoxesMidpts; shapeBBMidpt1; shapeBBMidpt2; shapeBBMidpt3; shapeBBMidpt4];
end

arrowHeads = [];
arrowTails = [];

for i = 1:size(arrowHeadsAll, 1)
    arrowShapeDistances = [];
    
    for j = 1:size(shapeBoundingBoxesMidpts, 1)
        arrowShapeDistance  = pdist([arrowHeadsAll(i, 1), arrowHeadsAll(i, 2); shapeBoundingBoxesMidpts(j, 1), shapeBoundingBoxesMidpts(j, 2)],'euclidean');
        arrowShapeDistances = [arrowShapeDistances; arrowShapeDistance];
    end
    
    [~, minidx] = min(arrowShapeDistances(:));
    arrowHeads = [arrowHeads;  shapeBoundingBoxesMidpts(minidx, :)];
    arrowTails = [arrowTails; arrowTailsAll(i, :)];
end

hold on;
outputFile = 'Arrows Heads and Tails.jpg';
outputLocation = fullfile(outputFolder, outputFile);
saveas(gcf, outputLocation);

%Generate rectangles,arrows,circles and diamonds

finalIm = ones(rows, cols);
figure; imshow(finalIm);

circleCentres = shapesCentroids(shapeCircle,:); %Centre of each circle
circleRadii = shapesPerimeters(shapeCircle,:)./(2*pi); %Radius of each circle
viscircles(circleCentres, circleRadii, 'Color', 'k');

arrow('Start', arrowTails, 'Stop', arrowHeads, 'Type', 'line', 'LineWidth', 2);

rectsBoundingBoxes = shapeBBs(shapeRectangale, :);

for i = 1:size(rectsBoundingBoxes, 1)
    rectangle('Position', [rectsBoundingBoxes(i,1) rectsBoundingBoxes(i,2)...
        rectsBoundingBoxes(i,3) rectsBoundingBoxes(i,4)], 'EdgeColor','k',...
    'LineWidth',3);
    hold on;
end

hold on;
diadsBoundingBoxes = shapeBBs(shapeDiamond, :);

for i = 1:size(diadsBoundingBoxes,1)
    patch([diadsBoundingBoxes(i,1)+ 0.5*diadsBoundingBoxes(i,3) diadsBoundingBoxes(i,1)+diadsBoundingBoxes(i,3) ...
        diadsBoundingBoxes(i,1)+0.5*diadsBoundingBoxes(i,3) diadsBoundingBoxes(i,1) ],...
        [diadsBoundingBoxes(i,2) diadsBoundingBoxes(i,2)+0.5*diadsBoundingBoxes(i,4) ...
        diadsBoundingBoxes(i,2)+diadsBoundingBoxes(i,4) diadsBoundingBoxes(i,2)+0.5*diadsBoundingBoxes(i,4) ], 'w', 'EdgeColor', 'k', 'LineWidth',3);
    hold on;
end

%Save the result
title('Result');
outputFile = 'Result.jpg';
outputLocation = fullfile(outputFolder, outputFile);
saveas(gcf, outputLocation);



