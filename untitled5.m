clc;
clear all;
close all;

% Load the image
I = imread("dog.jpg");

% Get dimensions
[height, width, dim] = size(I);
fprintf('Image Dimensions: Height = %d, Width = %d, Channels = %d\n', height, width, dim);

% Convert to grayscale
A = im2gray(I);

%%%% First Part: Display Original and Grayscale Images
figure('Name', 'Showing Images', 'NumberTitle', 'off');
subplot(2, 3, 1);
imshow(I);
title('Original Image');

subplot(2, 3, 2);
imshow(A);
title('Grayscale Image');

%%%% Histogram and Rotation
H = imhist(A); % Compute histogram
H1 = imrotate(A, 30); % Rotate image by 30 degrees

subplot(2, 3, 3);
imshow(H1);
title('Rotated Image (30 Degrees)');

subplot(2, 3, 4);
plot(H);
title('Histogram of Grayscale Image');

H2 = imhist(H1); % Compute histogram of the rotated image
subplot(2, 3, 5);
plot(H2);
title('Histogram of Rotated Image');

%%%% Adding Salt-and-Pepper Noise
salt_pepper_I = imnoise(A, 'salt & pepper', 0.02); % Add noise with density = 0.02

subplot(2, 3, 6);
imshow(salt_pepper_I);
title('Salt & Pepper Noise');

%%%% Proposal Filtering: Box Filter and Gaussian Filter
filtered_salt1 = imboxfilt(salt_pepper_I); % Apply box filter
filtered_salt2 = imgaussfilt(salt_pepper_I, 1); % Apply Gaussian filter (sigma = 1)

figure('Name', 'Noise Reduction', 'NumberTitle', 'off');
subplot(1, 3, 1);
imshow(salt_pepper_I);
title('Image with Salt & Pepper Noise');

subplot(1, 3, 2);
imshow(filtered_salt1);
title('Box Filter');

subplot(1, 3, 3);
imshow(filtered_salt2);
title('Gaussian Filter');

%%%% Binomial Filter
B = [1, 2, 1; 2, 4, 2; 1, 2, 1] / 16; % Normalize binomial filter kernel

% Apply the binomial filter
binomialFiltered = imfilter(salt_pepper_I, B, 'replicate');

figure('Name', '3rd Order Binomial Filter', 'NumberTitle', 'off');
subplot(1, 2, 1);
imshow(salt_pepper_I);
title('Salt & Pepper Noise');

subplot(1, 2, 2);
imshow(binomialFiltered);
title('Binomial Filter Applied');
