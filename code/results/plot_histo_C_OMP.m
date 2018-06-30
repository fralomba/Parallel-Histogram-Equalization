function [] = plot_histo_C_OMP()

results_seq = xlsread('sequential/seq_globale.xlsx');

%results_parallel_2th = xlsread('parallel/OMP/2_thread/OMP_2th.xlsx');
results_parallel_4th = xlsread('parallel/OMP/4_thread/OMP_4th.xlsx');
%results_parallel_8th = xlsread('parallel/OMP/8_thread/OMP_8th.xlsx');
%results_parallel_OMP_choice = xlsread('parallel/OMP/OMP_thread/OMP_choice.xlsx');

results_CUDA = xlsread('parallel/CUDA/CUDA_best.xlsx');

figure(1);
%plot(results_seq,'LineWidth',2);
title('speed up CUDA')
xticklabels({'0','100x100','200x200','400x400','1600x1600','3200x3200','6400x6400', '12800x12800'});
ylabel('Time (s)');
hold on;

speed_up_CUDA = results_seq/results_CUDA;
%speed_up_omp = results_seq/results_parallel_4th;

% plot(results_parallel_2th,'LineWidth',2);
% plot(results_parallel_4th,'LineWidth',2);
% plot(results_parallel_8th,'LineWidth',2);
% plot(results_parallel_OMP_choice,'LineWidth',2);
% plot(speed_up_omp,'LineWidth',2);
plot(speed_up_CUDA,'LineWidth',2);

%legend({'2 thread','4 thread', '8 thread', 'OMP choice'},'FontSize',12);
legend({'speed up CUDA'},'FontSize',12);

end

