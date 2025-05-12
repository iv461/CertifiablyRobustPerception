%% Evaluate STRIDE by loading problem instances from HDF5 and write data to csv.
%% Heng Yang, June 29, 2021

clc; clear; close all; restoredefaultpath

function problem = load_registration_from_HDF5(filename)
    %% Loads a problem from hdf5 file 
    problem.cloudA = h5read(filename, "/P")';
    problem.cloudB = h5read(filename, "/Q")';
    problem.N = size(problem.cloudA, 2);
    problem.R_gt = h5read(filename, "/R_gt")'; % Transpose the rotation (invert it) since apparently the conventions are not the same
    problem.t_gt= h5read(filename, "/t_gt");
    % add data to the problem structure
    problem.type        = 'point cloud registration';
    problem.inlier_mask = h5read(filename, "/inlier_mask");
    problem.nrOutliers  = problem.N - size(problem.inlier_mask, 1);
    problem.outlierIDs  = h5read(filename, "/outlier_ids");
    % note that the noiseBoundSq is important to ensure tight relaxation and
    % good numerical performance of the solvers. If noiseBound is too small,
    % typically the SDP solvers will perform worse (especially SDPNAL+)
    problem.noiseBound  = h5read(filename, "/params/eps");
    problem.noiseBoundSq = problem.noiseBound * problem.noiseBound;
    problem.translationBound  = h5read(filename, "/params/data_scale"); 
end

function infostride = eval_on_hdf5(N, outlier_rate, i, adversarial_suboptimality, eps_b)
    filename = sprintf("synthetic_data_N=%d_adversarial_suboptimality=%.1f_outlier_rate=%.1f_eps=%.1f_i=%d.h5", N, adversarial_suboptimality, outlier_rate, eps_b, i)
    hdf5_data_path = "/home/ivo/dev/certifiable_registration/test_data/"
    filename = hdf5_data_path + filename
    disp(filename)
    
    %% paths to dependencies
    spotpath    = '../spotless';
    stridepath  = '../STRIDE';
    manoptpath  = '../manopt';
    mosekpath   = '/home/ivo/mosek';
    sdpnalpath  = '/home/ivo/SDPNAL+v1.0';
    addpath('../utils')
    addpath('./solvers')
    %% choose if run GNC for STRIDE
    rungnc      = true;

    %% generate random point cloud registration problem
    problem.N                = N;
    problem.outlierRatio     = outlier_rate;
    problem = load_registration_from_HDF5(filename);

    %% generate SDP relaxation
    addpath(genpath(spotpath))
    SDP       = relax_point_cloud_registration_v4(problem,'checkMonomials',false);
    fprintf('\n\n\n\n\n')
    rmpath(genpath(spotpath))

    %% Solve using STRIDE
    % primal initialization using GNC
    if rungnc
        solution = gnc_point_cloud_registration(problem);
        v        = lift_pcr_v4(solution.R_est(:),...
                            solution.t_est,...
                            solution.theta_est,...
                            problem.translationBound);
        X0       = rank_one_lift(v);

        gnc.R_err = getAngularError(problem.R_gt,solution.R_est);
        gnc.t_err = getTranslationError(problem.t_gt,solution.t_est);
        gnc.time  = solution.time_gnc;
        gnc.f_est = solution.f_est;
        gnc.info  = solution;
    else
        X0       = [];
    end
    % Dual initialization using chordal SDP
    addpath(genpath(spotpath))
    chordalSDP       = chordal_relax_point_cloud_registration(problem);
    fprintf('\n\n\n\n\n')
    rmpath(genpath(spotpath))
    prob = convert_sedumi2mosek(chordalSDP.sedumi.At,...
                                chordalSDP.sedumi.b,...
                                chordalSDP.sedumi.c,...
                                chordalSDP.sedumi.K);
    addpath(genpath(mosekpath))
    time0   = tic;
    param.MSK_IPAR_INTPNT_MAX_ITERATIONS = 20;
    [~,res] = mosekopt('minimize info',prob,param);
    time_dualInit = toc(time0);
    [~,~,Schordal,~] = recover_mosek_sol_blk(res,chordalSDP.blk);
    S_assm           = pcr_dual_from_chordal_dual(Schordal);

    % STRIDE main algorithm
    addpath(genpath(stridepath))
    addpath(genpath(manoptpath))

    pgdopts.pgdStepSize     = 10;
    pgdopts.SDPNALpath      = sdpnalpath;
    pgdopts.maxiterPGD      = 5;
    % ADMM parameters
    pgdopts.tolADMM         = 1e-10;
    pgdopts.maxiterADMM     = 1e4;
    pgdopts.stopoptionADMM  = 0;

    pgdopts.rrOpt           = 1:3;
    pgdopts.rrFunName       = 'local_search_pcr_v4';
    rrPar.blk = SDP.blk; rrPar.translationBound = problem.translationBound;
    pgdopts.rrPar           = rrPar;
    pgdopts.maxiterLBFGS    = 1000;
    pgdopts.maxiterSGS      = 1000;
    pgdopts.S0              = S_assm;

    [outPGD,Xopt,yopt,Sopt]     = PGDSDP(SDP.blk, SDP.At, SDP.b, SDP.C, X0, pgdopts);
    rmpath(genpath(manoptpath))

    infostride              = get_performance_pcr(Xopt,yopt,Sopt,SDP,problem,stridepath);
    infostride.totaltime    = outPGD.totaltime + time_dualInit;
    infostride.time         = [outPGD.totaltime,time_dualInit];
    if rungnc
        infostride.gnc = gnc; 
        infostride.totaltime = infostride.totaltime + gnc.time;
        %infostride.time = [infostride.time, gnc.time];
    end

    fprintf('\n\n\n\n\n')
end 

function evaluate_and_save_result_table(i)
    N_values = [30]
    outlier_rate_values = [0., 0.3, .6, .8, .9]; 
    adversarial_suboptimalities = [0.] 
    instances = [0,1,2] 
    % Initialize an empty table to store results
    resultsTable = table();
    timestamp = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss');
    timestamp_str = string(timestamp);
    filename = "STRIDE_evaluation_results_" + timestamp_str + ".csv";
    % Loop through each combination of evaluation parameters
    for i = 1:length(N_values)
        for j = 1:length(outlier_rate_values)
            for k = 1:length(instances)
                for l = 1:length(adversarial_suboptimalities)
                    % Get current parameters
                    N = N_values(i);
                    outlier_rate = outlier_rate_values(j);
                    instance_i = instances(k)
                    adversarial_suboptimality=adversarial_suboptimalities(l)
                    eps_bound = .5
                    infostride = eval_on_hdf5(N, outlier_rate, instance_i, adversarial_suboptimality, eps_bound)
                    excution_time_ms = infostride.totaltime * 1000.
                    residual_angle_deg = infostride.R_err
                    residual_translation = infostride.t_err
                    num_iterations = 1
                    eta_suboptimality = infostride.Rs % It's called eta in get_performance_pcr
                    method = "STRIDE"
                    
                    % Append a new row to the table with the parameters and result
                    resultsTable = [resultsTable; table(N, outlier_rate, excution_time_ms, residual_angle_deg, residual_translation, num_iterations, eta_suboptimality, method)];
                    % Save regularly since this is a long-running-task
                    writetable(resultsTable, filename);
                    disp("Results saved to file: " + filename);
                end
            end
        end
    end

    % Display the table
    disp(resultsTable);
end

evaluate_and_save_result_table()
