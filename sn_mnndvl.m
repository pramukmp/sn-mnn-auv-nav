% Import necessary libraries
%addpath('/path/to/MemoryNetwork'); % Make sure to replace with the actual path

% Set random seed
rng(10);


% Define constants and parameters
neeta = 1e-3;
neeta_dash = 5e-4;
lipschitz_constant = 1.2;
epochs = 70;

% Create MemoryNeuralNetwork object
mnn = MemoryNeuralNetwork(10, 50, 3, neeta, neeta_dash, lipschitz_constant, true);

% Load data
path = fileparts(mfilename(''));
IMU_in = load(fullfile(path, 'dataset/TrainAndValidation/IMU.mat'));
V = load(fullfile(path, 'dataset/TrainAndValidation/V.mat'));
/home/airl/auvnav/sn-mnn-auv-nav
% DVL speed to beams - pitch=20 deg eqn 4 substitution
b1 = [cos((45 + 0 * 90) * pi / 180) * sin(20 * pi / 180), ...
    sin((45 + 0 * 90) * pi / 180) * sin(20 * pi / 180), cos(20 * pi / 180)];
b2 = [cos((45 + 1 * 90) * pi / 180) * sin(20 * pi / 180), ...
    sin((45 + 1 * 90) * pi / 180) * sin(20 * pi / 180), cos(20 * pi / 180)];
b3 = [cos((45 + 2 * 90) * pi / 180) * sin(20 * pi / 180), ...
    sin((45 + 2 * 90) * pi / 180) * sin(20 * pi / 180), cos(20 * pi / 180)];
b4 = [cos((45 + 3 * 90) * pi / 180) * sin(20 * pi / 180), ...
    sin((45 + 3 * 90) * pi / 180) * sin(20 * pi / 180), cos(20 * pi / 180)];

A = [b1; b2; b3; b4]
p_inv = (A' * A) \ A';
beams = zeros(length(V), 4);
for i = 1:length(V)
    beams(i, :) = (A*(V.data(:, i)) * (1 + 0.007))'; % scale factor 0.7%
end

beams_noise = beams + (0.042^2) * randn(length(V), 4) + 0.001 * ones(length(V), 4);

T = 100; % Sampling rate IMU - 100Hz, DVL - 1Hz
X_gyro = zeros(length(IMU_in.data(1, :, 1)/T), 3);
X_acc = zeros(length(IMU_in.data(1, :, 1) / T), 3);
Y = zeros(length(IMU_in.data(1, :, 1) / T), 4);
Z = zeros(3,length(V.data(1, :) / T));

n = 1;
V = V';

for t = 1:T:length(IMU_in.data(1, :, 1))

    X_acc(n, :) = IMU_in.data(:, t, 1);
    X_gyro(n, :) = IMU_in.data(:, t, 2);
    y = beams_noise(n, :);
    Y(n, :) = y; % DVL
    z = V.data(n, :); % velocities target variable
    Z(n, :) = z;
    n = n + 1;

end

% Num of DVL samples
N = length(IMU_in.data(1, :, 1) / T);

input_data = [X_acc, X_gyro, Y];
input_data_df = array2table(input_data);
disp(length(input_data));

error_sum = 0.0;
error_list = zeros(1, epochs);
rmse_list = zeros(1, epochs);
unstable_flag = false;
rmse_sum = 0.0;

try
    for i = 1:2000
        mnn.feedforward(zeros(10, 1));
        mnn.backprop(zeros(3, 1));
    end
    for epoch = 1:epochs
        pred_list = [];
        if epoch ~= 1
            fprintf('Training for epoch %2d finished with average loss of %.5f\n', epoch - 1, error_sum / length(input_data));
            error_list(epoch) = error_sum / length(input_data);
            rmse_list(epoch) = rmse_sum

 length(input_data);
        end
        error_sum = 0;
        rmse_sum = 0;
        fprintf('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n');

        for i = 2:length(input_data)
            if mnn.squared_error > 1e30
                unstable_flag = true;
                break;
            end
            pred = mnn.feedforward(input_data(i - 1, :)');
            mnn.backprop(Z(i - 1, :)');
            if epoch > 45
                pred_list = [pred_list; pred'];
            end
            error_sum = error_sum + mnn.squared_error;
            rmse_sum = rmse_sum + mnn.rmse;
            fprintf('Training for epoch %2d, progress %5.2f%% with squared loss: %.5f with rmse: %.5f\r', epoch, (i / length(input_data)) * 100, mnn.squared_error, mnn.rmse);
        end
    end
    fprintf('\n=====================================================================================================\n');

catch ME
    fprintf('%s', ME.message);
end

% Define RMSE function
function [rmse_ls, rmse_predicted, improv] = RMSE(true, predicted, LS)
    true = vecnorm(true, 2, 2);
    predicted = vecnorm(predicted, 2, 2);
    LS = vecnorm(LS, 2, 2);
    rmse_ls = sqrt(mean((true - LS).^2));
    rmse_predicted = sqrt(mean((true - predicted).^2));
    improv = 100 * (1 - (rmse_predicted / rmse_ls));
end

% Define MAE function
function [mae_ls, mae_predicted] = MAE(true, predicted, LS)
    true = vecnorm(true, 2, 2);
    predicted = vecnorm(predicted, 2, 2);
    LS = vecnorm(LS, 2, 2);
    mae_ls = sum(abs(LS - true)) / length(true);
    mae_predicted = sum(abs(predicted - true)) / length(true);
end

% Define NSE_R2 function
function [r2_ls, r2_predicted] = NSE_R2(true, predicted, LS)
    true = vecnorm(true, 2, 2);
    predicted = vecnorm(predicted, 2, 2);
    LS = vecnorm(LS, 2, 2);
    true_avg = mean(true);
    temp_ls = sum((LS - true).^2) / sum((true - true_avg).^2);
    r2_ls = 1 - temp_ls;
    temp_predicted = sum((predicted - true).^2) / sum((true - true_avg).^2);
    r2_predicted = 1 - temp_predicted;
end

% Define VAF function
function [r2_ls, r2_predicted] = VAF(true, predicted, LS)
    true = vecnorm(true, 2, 2);
    predicted = vecnorm(predicted, 2, 2);
    LS = vecnorm(LS, 2, 2);
    true_var = var(true);
    temp_ls = var(true - LS);
    r2_ls = (1 - temp_ls / true_var) * 100;
    temp_predicted = var(true - predicted);
    r2_predicted = (1 - temp_predicted / true_var) * 100;
end
