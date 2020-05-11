clc;
clear all;
train_data = dlmread("usps_train.txt", ",");
test_data = dlmread("usps_test.txt", ",");

function [train_x, train_y, ntrain, test_x, test_y, ntest] = usps(traindata, testdata)
  ntrain = rows(traindata);
  train_x = traindata(:,1:end-1);
  train_y = traindata(:, end);
  ntest = rows(testdata);
  test_x = testdata(:, 1:end-1);
  test_y = testdata(:, end);
endfunction

[train_x train_y ntrain test_x test_y ntest] = usps(train_data, test_data);

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

#step 1: Compute SVDs
maxrank = 35;
function [U3D] = createU3D(threeDtrain,count, max_rank)
  for i=0:9
    A = squeeze(threeDtrain(i+1, :, :));
    A = A(1:count(i+1), :)';
    [U S V] = svd(A);
    #size(A)
    U3D(i+1, :, :) = U(:, 1:max_rank);
  endfor
endfunction

U3D = createU3D(threeD, count, maxrank);

#step 2
function z = leastsqerror(Uk, testx, ntest)
  for j=1:ntest
    z(j) = norm((eye(256) - (Uk*Uk'))*testx(j, :)');
  endfor
endfunction

#step 3
function [lsk] = error(U3D, testx, ntest)
  for j=0:9
    Uk = squeeze(U3D(j+1,:, :));
    lsk(:, j+1) = leastsqerror(Uk, testx, ntest);
  endfor
endfunction

lsk = error(U3D, test_x, ntest);
size(lsk)

#step 4 - classify k=35
count_pre = 0;
for i=1:ntest
  min_value = 999999;
  min_digit = 0;
  for j=0:9
    if (lsk(i, j+1) < min_value)
      min_value = lsk(i, j+1);
      min_digit = j;
    endif
  endfor
  if min_digit == test_y(i)
    count_pre = count_pre+1;
  endif
endfor

display(count_pre)
display(ntest)
size(test_y)

#step 5 - classify k=1:35
arr = [];
for i=1:maxrank
  k = i;
  U3D = createU3D(threeD, count, k);
  lsk = error(U3D, test_x, ntest);
  count_pre = 0;
  for i=1:ntest
    min_value = 9999999;
    min_digit = -1;
    for j=0:9
      if (lsk(i, j+1) < min_value)
        min_value = lsk(i, j+1);
        min_digit = j;
      endif
    endfor
    if (min_digit == test_y(i))
      count_pre++;
    endif
  endfor
  display(count_pre)
  display(ntest)
  display(k)
  arr(i) = count_pre;
endfor

#step 6 - graphing
y = [];
for i=1:35
  y(i) = ntest - arr(i);
endfor
arr
y
x = linspace(1, 35);
plot(y);



















