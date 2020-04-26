clc;
clear all;
train_data = dlmread("usps_train.txt", ",");
test_data = dlmread("usps_test.txt", ",");

#problem 1
function [train_x, train_y, ntrain, test_x, test_y, ntest] = usps(traindata, testdata)
  ntrain = rows(traindata);
  train_x = traindata(:,1:end-1);
  train_y = traindata(:, end);
  ntest = rows(testdata);
  test_x = testdata(:, 1:end-1);
  test_y = testdata(:, end);
endfunction

[train_x train_y ntrain test_x test_y ntest] = usps(train_data, test_data);

#problem 2
function [IM] = converttwoD(rv)
  cols = columns(rv);
  IM = [];
  row_index = 1;
  col_index = 1;
  for i=1:256
    IM(row_index, col_index) = rv(i);
    col_index = col_index + 1;
    if (rem(i, 16) == 0)
      col_index = 1;
      row_index = row_index + 1;
    endif
  endfor
endfunction

IM1 = converttwoD(test_x(1, :));
IM2 = converttwoD(test_x(2, :));
IM3 = converttwoD(test_x(3, :));

figure 1;
imshow(50*IM1);
figure 2;
imshow(50*IM2);
figure 3;
imshow(50*IM3);

#problem 3
figure 4;
surf(IM1);
figure 5;
surf(IM2);
figure 6;
surf(IM3);

#problem 4
function [threeDtrain, count] = convertThreeD(train_x, train_y, ntrain)
  count = zeros(10, 1);
  for i=1:ntrain
    count(train_y(i, :)+1, :) = count(train_y(i, :)+1, :) + 1;
  endfor
  
  threeDtrain = zeros(10, max(count), 256);
  for i=1:10
    digit_array = [];
    for j=1:ntrain
      if (train_y(j) == i-1)
        digit_array = [digit_array; train_x(j, :)];
      endif
    endfor
    threeDtrain(i, 1:count(i), :) = digit_array;
  endfor
endfunction

[threeD, count] = convertThreeD(train_x, train_y, ntrain);

#problem 5
function centroids = findCentroids(threeDdata, digit_count)
  centroids = zeros(10, 256);
  for i=1:10
    centroids(i, :) = sum(threeDdata(i, :, :), 2);
    centroids(i, :) = (1/digit_count(i))*centroids(i, :);
  endfor
endfunction

centroids = findCentroids(threeD, count);
display(centroids);

figure 7;
#digit 3
im = converttwoD(centroids(8, :));
imshow(im)


#test
correct_predictions = 0;
for i=1:ntest
  min_digit = 0;
  min_val=999999;
  for j=1:10
    eucl_dist = norm(test_x(i, :) - centroids(j, :), 2);
    if (eucl_dist < min_val)
      min_val = eucl_dist;
      min_digit = j-1;
    endif
  endfor
  min_digit
  test_y(i, 1)
  if (min_digit == test_y(i, 1))
    correct_predictions = correct_predictions + 1;
  endif
endfor

correct_predictions
display("out of 2007 test cases predicted");
  


