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
  display(count);
  display(size(train_x));
  threeDtrain = zeros(10, max(count), 256);
  for i=1:ntrain
    num = train_x(i, 1) + 1;
    threeDtrain(num, 1:count(num), 256) = train_x(i, :);
  endfor
endfunction

[three, count] = convertThreeD(train_x, train_y, ntrain);



  


