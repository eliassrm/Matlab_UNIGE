clear 
close all
clc

% Loading trajectory 9 of testing, which is very normal
trajectoryNum = 9;
load dataFollowerTest
load abn_test_follower_traj10

% Anomalies from MJPF
abnormMeas = estimationAbn.mean_error;
abnormdCLA = estimationAbn.CLA;
abnormdCLB = estimationAbn.CLB;
KLDabn_all = estimationAbn.sommaKLD_simmetrica;

% Anomalies from UMKF
abnUMKF = data.MMCell{1, trajectoryNum}(:, 3:4);
abnUMKF_meaned = mean(abs(abnUMKF), 2);

subplot(5,1,1);
cla
plot(abnUMKF_meaned(2:end),'-b')
title('Averaged error (UMKF)')
grid on
subplot(5,1,2);
cla
plot(abnormMeas(2:end),'-r')
title('Averaged error (MJPF)')
grid on
subplot(5,1,3);
cla
plot(abnormdCLA(2:end),'-r')
title('CLA (MJPF)')
grid on
subplot(5,1,4);
cla
plot(abnormdCLB(2:end), '-r')
title('CLB (MJPF)')
grid on
subplot(5,1,5);
cla
plot(KLDabn_all(2:end),'-b')
title('KLDA (MJPF)')
grid on