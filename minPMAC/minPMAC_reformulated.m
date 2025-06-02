function [FEAS_FLAG, bu_a, info] = minPMAC_reformulated(H, bu_min, w, cb)

    % start timer
    tic;

    % bu_min =  1/(3-cb) * bu_min;
    bu_min = cb * bu_min;
    [Ly, U, N] = size(H);
    Lx = ones(1, U);
    index_end = cumsum(Lx);
    index_start = [1,index_end(1:end-1)+1];
    
    
    subsets = powerSet(1:U);
    subsets = subsets(2:end);
    tic
    Rxx = cell(U, N);

    disp("Beginning initial CVX for energy optimization and decoding order");

    cvx_begin quiet
        cvx_solver mosek
        variable Rxx(Lx(1), Lx(1), U, N) hermitian semidefinite
        variable b(U, N) nonnegative
        dual variable theta
        expressions rate_sum
    %     for u = 1:U
    %         for n = 1:N
    %             Rxx(:, :, u, n) == semidefinite(Lx(1));
    %         end
    %     end
        theta : sum(b, 2)' >= bu_min;
        
    %     for u = 1:U
    %         for n = 1:N
    %             b(u, n) >= 0;
    %         end
    %     end
        
        % for u = 1:U
        %     for n = 1:N
        %         for i = 1:Lx(1)
        %             Rxx(i, i, u, n) >= 0;
        %         end
        %     end
        % end
        
         for subset = subsets
             for n = 1:N
                 temp = zeros(Ly, Ly);
                 for u = subset{:}
                     temp = temp + H(:, u, n) * Rxx(:, :, u, n) * H(:, u, n)';
                 end
%                  for u = subset{:}
%                     temp = temp + H(:,index_start(u):index_end(u), n) * Rxx(:, :, u, n) * H(:,index_start(u):index_end(u), n)';
%                  end
    
    
                 capacity_bound = log_det(eye(Ly) + temp)/log(2);
                 rate_sum = 0;
                 for u = subset{:}
                    rate_sum = rate_sum + b(u, n);
                 end
                 rate_sum <= capacity_bound;
             end
         end
         obj = 0;
         for u = 1:U
             for n =1:N
                obj = obj + w(u)*trace(Rxx(:,:, u, n));
             end
         end
         minimize (obj)
    cvx_end

    disp("Rxx is");
    disp(Rxx);

    disp("cvx finished");
    disp("Status is");
    disp(cvx_status);

    % calculate energies
    Eu_a = zeros(1, U);
    for u=1:U
        for n=1:N
            Eu_a(u) = Eu_a(u) + trace(reshape(Rxx(1:Lx(u), 1:Lx(u), u, n), Lx(u), Lx(u)));
        end
    end
    Eu_a = Eu_a/N;

    % bu_min = (3-cb) * bu_min;
    % b = (3-cb) * b;
    bu_min = 1/cb * bu_min;
    b = 1/cb * b;

    % disp(theta)

    % theta = theta';

    % temporary - remove later
    % disp("Rates from initial cvx");
    % disp(sum(b, 2)');

    % Cluster users by theta values
    [clusters, unique_theta, theta] = identify_clusters(theta);
    unique_theta = unique_theta';

    disp("unique theta is");
    disp(unique_theta);
    disp("clusters is");
    disp(clusters);

    theta = theta';
    
    % Generate all possible orders considering clusters
    all_orders = generate_all_orders(clusters);
    
    disp("all order are");
    disp (all_orders);

    flag_time_sharing_required = 0;
    if (size(all_orders, 1) > 1)
        % only one order exists, no time sharing
        flag_time_sharing_required = 1;
    else
        FEAS_FLAG = 1;
    end
        
    % Find optimal weights for these orders
    if (flag_time_sharing_required==1)
        disp("Starting time sharing");
        [weights, orderings, Rxx_orders, Eun_orders, bun_orders, bu_a_orders, num_orders] = find_optimal_weights(H, Rxx, all_orders, unique_theta, bu_min, w, Ly, U, N, cb);
        disp("Time sharing finished");
        FEAS_FLAG = sum(weights > 0) > 0;  % Feasibility flag 0-infeasible, 1-feasible
    end
    
    if (FEAS_FLAG == 1 && flag_time_sharing_required == 1)
        if (sum(weights > 0)) > 1  % Feasibility flag 1-no time sharing, 2-time sharing
            FEAS_FLAG = 2;
        end
    end
    
    if (flag_time_sharing_required == 1)
        % calculate bu_a from bu_a_orders
        weights = weights';
        weights_reshaped = reshape(weights, [1, num_orders]);
        weights_reshaped = repmat(weights_reshaped, [U, 1]);
        bu_a = sum(bu_a_orders .* weights_reshaped, 2);
        bu_a = bu_a';
    else
        [b_achieved, bu_a, Eun] = compute_achieved_rates_order_given(H, Rxx, all_orders, Ly, U, N, cb);
    end
    
    % Store information
    info.H = H;
    info.bu_min = bu_min;
    info.w = w;
    info.cb = cb;
    info.unique_theta = unique_theta;
    info.theta = theta;
    info.clusters = clusters;
    if (flag_time_sharing_required == 1)
        info.orderings = orderings;
        info.time_sharing_weights = weights;
        info.Rxx_orders = Rxx_orders;
        info.Eun_orders = Eun_orders;
        info.bun_orders = bun_orders;
        info.bu_a_orders = bu_a_orders;
        info.bu_a = bu_a;
    else
        info.Rxx = Rxx;
        info.Eun = Eun;
        info.bun = b_achieved;
        info.bu_a = bu_a;
    end
    info.FEAS_FLAG = FEAS_FLAG;
    info.Eu_a = Eu_a;

    elapsedTime = toc;
    disp(['Elapsed time for reformulated minPMAC: ', num2str(elapsedTime), ' seconds']);
end

% 
function [weights, orderings, Rxx_orders, Eun_orders, bun_orders, bu_a_orders, num_orders] = find_optimal_weights(H, Rxx, all_orders, theta, bu_min, w, Ly, U, N, cb)
    num_orders = size(all_orders, 1);
    % disp(num_orders);
    orderings = cell(num_orders, 1);
    Rxx_orders = cell(num_orders, 1);
    Eun_orders = zeros(U, N, num_orders);
    bu_a_orders = zeros(U, num_orders);
    b_achieved_orders = zeros(U, N, num_orders);

    % Compute the achieved rates for each ordering
    for i = 1:num_orders
        orderings{i} = all_orders(i, :);
        [b_achieved_orders(:, :, i), bu_a_orders(:,i), Eun_orders(:,:,i)] = compute_achieved_rates_order_given(H, Rxx, orderings{i}, Ly, U, N, cb);
    end

    disp("b_achieved_orders is");
    disp(b_achieved_orders);

    % Solve for the optimal weights using mixed-integer linear programming
    tolerance = 1e-7;
    cvx_begin quiet
        cvx_solver mosek
        variable weights(num_orders) nonnegative
        variable z(num_orders) binary
        minimize(sum(z))  % Minimize the number of non-zero weights
        subject to
            weight_sum = sum(weights);
            weight_sum == 1;
            % disp(size(bu_min));
            % disp(size(sum(sum(b_achieved_orders .* reshape(weights, [U  num_orders]), 3), 2)'));
            abs(sum(sum(b_achieved_orders .* repmat(reshape(weights, [1 1 num_orders]), [U, N, 1]), 3), 2)' - bu_min) <= tolerance;
            weights <= z;  % Ensure weights are zero where z is zero
    cvx_end

    % Store the results
    for i = 1:num_orders
        if weights(i) > 0
            Rxx_orders{i} = Rxx;
        end
    end
    % correct the above - obviously wrong
    
    bun_orders = b_achieved_orders;

end

function [b_achieved, bu_a, Eun] = compute_achieved_rates_order_given(H, Rxx, ...
        order, Ly, U, N, cb)
    % order given
    pi_inv = order;
    
    % Initialize b_achieved
    b_achieved = zeros(U, N);
    
    % Calculate b_achieved
    for n = 1:N
        for u = 1:U
            mat1 = 0;
            for i = u:U
                % disp(size(H));
                % disp(n);
                % disp(pi_inv(i));
                % disp(size(H(:, pi_inv(i), n)));
                % disp ("errata");
                % disp(size(Rxx(:, :, pi_inv(i), n)));
                mat1 = mat1 + H(:, pi_inv(i), n) * Rxx(:, :, pi_inv(i), n) * H(:, pi_inv(i), n)';
            end
            mat2 = 0;
            for i = u+1:U
                mat2 = mat2 + H(:, pi_inv(i), n) * Rxx(:, :, pi_inv(i), n) * H(:, pi_inv(i), n)';
            end
            
            b_achieved(pi_inv(u), n) = log2((det(eye(Ly) + mat1))) - log2((det(eye(Ly) + mat2)));
        end
    end

    b_achieved = b_achieved / cb;

    % Compute bu_a
    bu_a = sum(b_achieved, 2);
    
    % Compute Eun
    Eun = zeros(U, N);
    for u = 1:U
        for n = 1:N
            Eun(u, n) = trace(Rxx(:, :, u, n));
        end
    end
end

function pSet = powerSet(S)
    n = length(S);            
    numSubsets = 2^n;         
    pSet = cell(1, numSubsets); 

    for i = 0:numSubsets-1
        binaryIndex = bitget(i, n:-1:1);  
        subset = S(logical(binaryIndex)); 
        pSet{i+1} = subset;               
    end
end

function all_orders = generate_all_orders(clusters)
    % Generate all permutations of users considering clusters
    all_orders = perms(clusters{1});
    for i = 2:length(clusters)
        new_orders = perms(clusters{i});
        all_orders = combvec(all_orders, new_orders);
    end
    if length(clusters) > 1
        all_orders = all_orders';
    end
end

function [clusters, unique_theta, theta] = identify_clusters(theta)
        % Find unique thetas
        % Define the tolerance
        tolerance = 1e-5;
        
        % Sort the theta values
        sorted_theta = sort(theta);
        
        % Initialize the unique_theta array with the first value
        unique_theta = sorted_theta(1);
        
        % Loop through the sorted theta values and check the difference
        for i = 2:length(sorted_theta)
            if abs(sorted_theta(i) - unique_theta(end)) > tolerance
                unique_theta = [unique_theta; sorted_theta(i)];
            end
        end
        % disp("Size of unique theta is");
        % disp(size(unique_theta));
        
        % Initialize clusters as a cell array
        clusters = cell(length(unique_theta), 1);
        
        % Assign users to clusters based on their theta values
        for i = 1:length(unique_theta)
            clusters{i} = find(abs(theta - unique_theta(i)) < tolerance);
        end
        
        % Display the clusters and their corresponding theta values
        % disp('Clusters and their corresponding theta values:');
        % for i = 1:length(clusters)
        %     fprintf('Theta value: %f', unique_theta(i));
        %     % disp(clusters{i});
        % end
    end

function [b_achieved, bu_a, Eun] = compute_achieved_rates(H, Rxx, ...
        theta, Ly, U, N)
    % Sort theta and compute pi and pi_inv
    [~, pi_inv] = sort(theta);
    [~, pi] = sort(pi_inv);
    
    % Initialize b_achieved
    b_achieved = zeros(U, N);
    
    % Calculate b_achieved
    for n = 1:N
        for u = 1:U
            mat1 = 0;
            for i = u:U
                mat1 = mat1 + H(:, pi_inv(i), n) * Rxx(:, :, pi_inv(i), n) * H(:, pi_inv(i), n)';
            end
            mat2 = 0;
            for i = u+1:U
                mat2 = mat2 + H(:, pi_inv(i), n) * Rxx(:, :, pi_inv(i), n) * H(:, pi_inv(i), n)';
            end
            
            b_achieved(pi_inv(u), n) = log2((det(eye(Ly) + mat1))) - log2((det(eye(Ly) + mat2)));
        end
    end
    
    % Transform theta
    theta = theta';
    
    % Compute bu_a
    bu_a = sum(b_achieved, 2);
    
    % Compute Eun
    Eun = zeros(U, N);
    for u = 1:U
        for n = 1:N
            Eun(u, n) = trace(Rxx(:, :, u, n));
        end
    end
end