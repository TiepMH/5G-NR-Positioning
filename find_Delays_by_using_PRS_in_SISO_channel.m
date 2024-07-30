clear all; clc;

n_TxAnt = 1; % number of Tx antennas
n_RxAnt = 1; % number of Rx antennas
n_gNBs = 4; % number of gNBs

%% Carriers
% Create carrier-objects
carriers = repmat(nrCarrierConfig, ...
                  1, n_gNBs); % a cell with (1 x n_gNBs) elements each being an object
% Configure carrier properties
cell_IDs = [1,   2,   3,   4];
for idx = 1:n_gNBs
    % different IDs
    carriers(idx).NCellID = cell_IDs(idx);
end
% Since all carrier-objects are created by nrCarrierConfig, ...
% we only have to get OFDM information from one carrier-object.
OFDM_Info = nrOFDMInfo(carriers(1));
Nfft = OFDM_Info.Nfft;
SampleRate = OFDM_Info.SampleRate;
SlotsPerFrame = OFDM_Info.SlotsPerFrame; % number of slots per frame
% Look at if numerology is 0, or 1 or 2 ...
SubcarrierSpacing = carriers(1).SubcarrierSpacing; % in kHz
if SubcarrierSpacing == 15 % default
    numerology = 0;
elseif SubcarrierSpacing == 30
    numerology = 1;
elseif SubcarrierSpacing == 60
    numerology = 2;
elseif SubcarrierSpacing == 120
    numerology = 3;
elseif SubcarrierSpacing == 240
    numerology = 4;
elseif SubcarrierSpacing == 480
    numerology = 5;
elseif SubcarrierSpacing == 960
    numerology = 6;
end

%% PRS Configuration
% Create PRS-objects
PRSs = repmat(nrPRSConfig, ...
              1, n_gNBs); % a cell with (1 x n_gNBs) elements each being an object

% Configure PRS properties
PRS_SlotOffsets = [0,   2,   4,   7];
PRS_IDs = [11,   12,   13,   14]; % from 0 (default) to 4095

TPRSPeriod = (2^numerology)*10; % 2^numerology is multiplied by one value in the set {10, 16, 20, 32, ... 10240}
for idx = 1:n_gNBs
    % same properties
    PRSs(idx).PRSResourceSetPeriod = [TPRSPeriod 0]; % 'on' (default), 'off', or 2-element vector
    PRSs(idx).NumRB = 51; % from 0 to 275
    PRSs(idx).CombSize = 4; % 2 (default) or 4, 6, 12
    PRSs(idx).NumPRSSymbols = 10; % from 0 to 12 (default)
    % different IDs
    PRSs(idx).NPRSID = PRS_IDs(idx);
    % different time offsets
    PRSs(idx).PRSResourceOffset = PRS_SlotOffsets(idx);
end

%% PDSCH Configuration
PDSCH = nrPDSCHConfig;
PDSCH.PRBSet = 0:51;
PDSCH.SymbolAllocation = [0 14];
PDSCH.DMRS.NumCDMGroupsWithoutData = 1; % quan trong
PDSCH = repmat(PDSCH,1,n_gNBs);

%% Generate PRS resource grid and PDSCH resource grid
% number of frames and number of slots
n_Frames = 1; % each frame is 10 ms long
total_Slots = n_Frames*SlotsPerFrame;

%% PRS resource grid
PRS_grids = cell(1,n_gNBs); % to store the resultant PRS grids for gNBs 
slots_containing_PRS_symbols = []; % to identify which slots contain PRS symbols
for idx = 1:n_gNBs
    % Carrier object
    carrier_i = carriers(idx); % Carrier for a specific gNB
    % PRS object
    PRS_i = PRSs(idx);
    % Initialize the PRS grid (for the i-th gNB) by an empty object
    PRS_grid_i = []; % Let's call it "the i-th PRS grid"

    % At each slot, we repeat the process of generating PRS symbols and indices
    for slot_k = 0:total_Slots-1
        % UPDATE the slot index for this carrier object
        carrier_i.NSlot = slot_k; 

        % PRS symbols and indices at a specific slot
        PRS_symbols = nrPRS(carrier_i, PRS_i);
        PRS_indices = nrPRSIndices(carrier_i, PRS_i);
        
        % Create a resource grid spanning ONLY 1 slot in time domain
        grid_of_1_slot = nrResourceGrid(carrier_i, n_TxAnt); % size = (?, SymbolsPerSlot)

        % Map PRS symbols to the grid
        grid_of_1_slot(PRS_indices) = PRS_symbols;

        % Over time, "the i-th PRS grid" will grow by 1 slot
        PRS_grid_i = [PRS_grid_i grid_of_1_slot]; % expanded by 1 slot
        
        % Identify specific slots that contain PRS symbols
        if ~isempty(PRS_symbols)
            slots_containing_PRS_symbols = [slots_containing_PRS_symbols, slot_k];
        end % end if

    end % looping-over-slots ends

    % After an expanding process over slots, "the i-th PRS grid" is now complete
    PRS_grids{idx} = PRS_grid_i; % STORE "the i-th PRS grid" for the i-th gNB 

end % looping-over-gNBs ends 

%% Data resource grid
data_grids = cell(1,n_gNBs); % to store the resultant data grids for gNBs 
for idx = 1:n_gNBs
    % Carrier object
    carrier_i = carriers(idx); % Carrier for a specific gNB
    % PDSCH object
    PDSCH_i = PDSCH(idx);
    % Initialize the PDSCH grid (for the i-th gNB) by an empty object
    data_grid_i = []; % Let's call it "the i-th data grid"

    for slot_k = 0:total_Slots-1
        % Create a resource grid spanning ONLY 1 slot in time domain
        grid_of_1_slot = nrResourceGrid(carrier_i, n_TxAnt); % size = (?, SymbolsPerSlot)           

        % If slot_idx is NOT a member of slots_containing_PRS_symbols,
        % then this slot_idx does NOT contain any PRS symbol.
        % We can make use of this available slot to transmit data.
        if ~ismember(slot_k, slots_containing_PRS_symbols)
            % Generate PDSCH indices
            [pdschInd,pdschInfo] = nrPDSCHIndices(carrier_i, PDSCH_i);

            % Generate random data bits for transmission
            data = randi([0 1],pdschInfo.G,1);
            % Generate PDSCH symbols
            pdschSym = nrPDSCH(carrier_i, PDSCH_i, data);

            % Generate demodulation reference signal (DM-RS) indices and symbols
            dmrsInd = nrPDSCHDMRSIndices(carrier_i, PDSCH_i);
            dmrsSym = nrPDSCHDMRS(carrier_i, PDSCH_i);

            % Map PDSCH and its associated DM-RS to slot grid
            grid_of_1_slot(pdschInd) = pdschSym;
            grid_of_1_slot(dmrsInd) = dmrsSym;
        end

        % Over time, "the i-th data grid" will grow by 1 slot
        data_grid_i = [data_grid_i grid_of_1_slot]; % expanded by 1 slot

    end % looping-over-slots ends

    % After an expanding process over slots, "the i-th data grid" is now complete
    data_grids{idx} = data_grid_i; % STORE "the i-th data grid" for the i-th gNB 

end % looping-over-gNBs ends

%% Plot PRS and data resource grids
plot_PRS_and_PDSCH_grids(PRS_grids, data_grids)

%% Positions of gNBs in (x,y,z)-coordinate system
% UE position
UE_pos = [500 -20 1.8]; % 1.5m <= UE height <= 22.5m for 'UMa' path loss scenario
                       % see TR 38.901 Table 7.4.1-1
% Positions of gNBs
% height=25m for 'UMa' path loss scenario, see TR 38.901 Table 7.4.1-1 
gNB_pos{1} = 1e3*[2,    -5,    0.0250]; 
gNB_pos{2} = 1e3*[3,     4,    0.0250];
gNB_pos{3} = 1e3*[-6,    8,    0.0250];
gNB_pos{4} = 1e3*[-9,   -1,    0.0250];

% Distances from gNBs to UE
distances = zeros(1,n_gNBs);
for idx=1:n_gNBs
    distances(idx) = sqrt(sum(abs(gNB_pos{idx}-UE_pos).^2));
end

%% Path Loss Configuration
PathLoss = nrPathLossConfig;
PathLoss.Scenario = 'Uma';
fc = 3e9;    % Carrier frequency (Hz)
PLs = cell(1, n_gNBs);
for idx = 1:n_gNBs
    % Calculate path loss for each gNB and UE pair
    line_of_sight = true; % There is the line of sight (LOS) component
    PLdB = nrPathLoss(PathLoss, fc, ...
                      line_of_sight, ...
                      reshape(gNB_pos{idx},3,1), ...
                      reshape(UE_pos,3,1) ...
                      );
    PL = 10^(PLdB/10);
    PLs{idx} = PL;
end

%% On the Tx side:
txWaveform = cell(1,n_gNBs);
for idx = 1:n_gNBs
    % Carrier object
    carrier_i = carriers(idx); % Carrier for a specific gNB
    % carriers(idx).NSlot = 0; % This does not matter.
    % Perform OFDM modulation of both PRS signals and PDSCH signals
    txWaveform{idx} = nrOFDMModulate(carriers(idx), ...
                                     PRS_grids{idx} + data_grids{idx});
end

% Calculate the length of the transmitted samples
length_of_transmitted_samples = length(txWaveform{1});

%% Delays
LightSpeed = physconst('LightSpeed');
delays_in_samples = zeros(1,n_gNBs);
for idx = 1:n_gNBs
   distance = distances(idx);
   delay_in_seconds = distance/LightSpeed; % Delay of the i-th gNB in seconds
   delay_in_samples = round(delay_in_seconds*SampleRate); % Delay of the i-th gNB in samples
   delays_in_samples(idx) = delay_in_samples; % Store this value
end

% Find the maximum delay in samples
delay_max = max(delays_in_samples);

%% On the Rx side:
length_of_received_samples = length_of_transmitted_samples + delay_max;
rxWaveform = zeros(length_of_received_samples, 1);

for idx = 1:n_gNBs
    % Calculate path loss for each (gNB, UE) pair
    PL = PLs{idx};
    % Delay of the i-th gNB in samples
    delay_in_samples = delays_in_samples(idx);

    % Let's consider the contribution from a specific gNB:
    % First, delay the transmitted samples  
    % Second, take the path loss into account
    contribution_from_gNB_i = (1/sqrt(PL))*[zeros(delay_in_samples, 1); 
                                            txWaveform{idx}; ...
                                            zeros(delay_max - delay_in_samples, 1)
                                            ];

    % Sum up all the contributions from all gNBs
    rxWaveform = rxWaveform + contribution_from_gNB_i;
end

size(rxWaveform)

%% Estimate time delays
corrs = cell(1,n_gNBs);
delays_est = zeros(1,n_gNBs);
peaks = zeros(1,n_gNBs);
for idx = 1:n_gNBs
    % CROSS-CORRELATION is used to estimate the time delay:
    % In particular, we cross-correlate the input waveform and the reference waveform
    % The input waveform is the 1st parameter
    % The reference waveform is obtained by modulating the ref grid (the 3rd parameter)
    % Matlab function: [offset,mag] = nrTimingEstimate(carrier,waveform,refGrid)
    [offset,mag] = nrTimingEstimate(carriers(idx), ...
                                    rxWaveform, ...
                                    PRS_grids{idx});

    % Ignore noisy side lobe peaks
    corrs{idx} = mag(1:(Nfft*SubcarrierSpacing/15));

    % The estimated delays are the peaks
    peaks(idx) = max(corrs{idx});
    delays_est(idx) = find(corrs{idx} == peaks(idx),1) - 1;    
end

%% Find 3 gNBs from the closest gNB to the farthest gNB, based on correlation values
[~, gNBs_from_closest_one] = sort(peaks, 'descend');
gNBs_from_closest_one = gNBs_from_closest_one(1:3);

% Plot PRS correlation results
plotPRSCorr(corrs, SampleRate);

%% Display results
disp(['Distances: ', num2str(distances)]);
disp(['Actual delays (in samples) : ', num2str(delays_in_samples)]);
disp(['Estimated delays (in samples) : ', num2str(delays_est)]);
disp(['3 closest gNBs (to be estimated) : ', num2str(gNBs_from_closest_one)]);


%% Functions
function plot_PRS_and_PDSCH_grids(PRS_grids, data_grids)
    PRS_grids_many_gNBs = zeros(size(PRS_grids{1})); % initialize a grid with background=0
    data_grids_many_gNBs = zeros(size(data_grids{1})); % initialize a grid with background=0
    
    % NOTE:
    %
    % PRS_grids is different from PRS_grids_many_gNBs.
    % PRS_grids contains grids of complex-valued symbols.
    %
    % By contrast, PRS_grids_many_gNBs is for illustration.
    % PRS_grids_many_gNBs will contain grids of numbers, e.g., integers
    % Each different number corresponds to a different color.
    %
    % Also, data_grids is different from data_grids_many_gNBs

    n_gNBs = numel(PRS_grids);
    my_legends = cell(1,n_gNBs);
    for idx=1:n_gNBs
        % PRS grid
        grid_of_abs_values_for_PRS = abs(PRS_grids{idx});
        grid_of_abs_values_for_PRS(grid_of_abs_values_for_PRS ~= 0) = idx;
        PRS_grids_many_gNBs = PRS_grids_many_gNBs + grid_of_abs_values_for_PRS;
        % data grid
        data_grids_many_gNBs = data_grids_many_gNBs + abs(data_grids{idx});
        % for legend
        my_legends{idx} = ['PRS from gNB' num2str(idx)];
    end
    % data grid: its color value > the color values of PRS grid
    data_grids_many_gNBs(data_grids_many_gNBs ~= 0) = n_gNBs+1;
    
    my_legends = [{' '}, my_legends, {'Data from all gNBs'}];
    num_colors = 1 + n_gNBs + 1; % background + colors for gNBs + color for data
    
    % resource grid containing (PRS + data) from many gNBs
    prsGrid_and_dataGrid_many_gNBs = PRS_grids_many_gNBs + data_grids_many_gNBs;
    
    % uisetcolor is a color picker
    % define the colormap by a 3-column matrix of RGB
    cmap = jet(num_colors);
    cmap(1,:) = [1 1 1]; % white background for prsGrid_many_gNBs
    cmap(2,:) = [0 0 1]; % blue color for PRS from gNB 1
    cmap(3,:) = [0 1 0]; % green color for PRS from gNB 2
    cmap(4,:) = [1 0 1]; % magenta color for PRS from gNB 3
    cmap(5,:) = [0 1 1]; % cyan color for PRS from gNB 4
    cmap(6,:) = [1 0.4 0.2]; % orange color for PRS from gNB 5
    cmap(end,:) = [0 0 0]; % black color for data
    
    % Plot both PRS and data resource grids
    figure()
    imagesc( prsGrid_and_dataGrid_many_gNBs )
    colormap(cmap);
    hold on
    L = line(ones(num_colors),ones(num_colors), 'LineWidth',2); 
    set(L,{'color'},mat2cell(cmap,ones(1,num_colors),3)); 
    %
    legend(my_legends{:});
    title('Resource Grid for PRS and PDSCH');
    xlabel('OFDM symbol');
    ylabel('Subcarrier');
    axis xy
    
    % Plot PRS resource grid
    figure()
    imagesc( PRS_grids_many_gNBs )
    colormap(cmap(1:end-1,:));
    hold on
    L = line(ones(num_colors-1),ones(num_colors-1), 'LineWidth',2); 
    set(L,{'color'},mat2cell(cmap(1:end-1,:),ones(1,num_colors-1),3)); 
    %
    legend(my_legends{1:end-1});
    title('Resource Grid for PRS');
    xlabel('OFDM symbol');
    ylabel('Subcarrier');
    axis xy
    
    % Plot PDSCH (data) resource grid
    figure()
    imagesc( data_grids_many_gNBs )
    colormap(cmap);
    hold on
    L = line(ones(1),ones(1), 'LineWidth',2); 
    set(L,{'color'},mat2cell(cmap(end,:),ones(1,1),3)); 
    %
    legend(my_legends{end});
    title('Resource Grid for PDSCH');
    xlabel('OFDM symbol');
    ylabel('Subcarrier');
    axis xy
end


function plotPRSCorr(corrs, SampleRate)
    n_gNBs = numel(corrs);
    num_colors = 1 + n_gNBs; % background + colors for gNBs

    % uisetcolor is a color picker
    % define the colormap by a 3-column matrix of RGB
    cmap = jet(num_colors);
    cmap(1,:) = [1 1 1]; % white background for prsGrid_many_gNBs
    cmap(2,:) = [0 0 1]; % blue color for PRS from gNB 1
    cmap(3,:) = [0 1 0]; % green color for PRS from gNB 2
    cmap(4,:) = [1 0 1]; % magenta color for PRS from gNB 3
    cmap(5,:) = [0 1 1]; % cyan color for PRS from gNB 4
    cmap(6,:) = [1 0.4 0.2]; % orange color for PRS from gNB 5

    % Line widths and 
    LineWidths = [1, 1, 1, 1, 1];
    LineStyles = ['-', '-', '-', '-', '-'];
    Makers = ['o', "square", '*', "diamond", '>'];
    
    % Plot correlation for gNBs
    figure
    time_in_seconds = (0:length(corrs{1}) - 1)/SampleRate;
    my_legends = cell(1,2*n_gNBs);
    for idx = 1:n_gNBs
        plot(time_in_seconds, abs(corrs{idx}), ...
            'Color', cmap(idx+1,:), ...
            'LineWidth', LineWidths(idx), ...
            'LineStyle', LineStyles(idx)); % Correlations
        my_legends{idx} = sprintf('gNB%d', idx);
        hold on
    end
    %
    for idx = 1:n_gNBs
        corr_abs = abs(corrs{idx});
        peak_positions = find(corr_abs == max(corr_abs), 1);
        plot(time_in_seconds(peak_positions), corr_abs(peak_positions), ...
            'Marker', Makers(idx),...
            'Color', cmap(idx+1,:), ...
            'LineWidth', 1 ...
            ); % Peaks
        my_legends{n_gNBs+idx} = '';
        hold on
        text(time_in_seconds(peak_positions), corr_abs(peak_positions), strcat('  gNB',num2str(idx)));
    end
    legend(my_legends);
    xlabel('Time (seconds)');
    ylabel('Absolute Value');

    figure
    % Plot correlation for gNBs
    samples = (0:length(corrs{1}) - 1);
    my_legends = cell(1,2*n_gNBs);
    for idx = 1:n_gNBs
        plot(samples, abs(corrs{idx}), ...
            'Color', cmap(idx+1,:), ...
            'LineWidth', LineWidths(idx), ...
            'LineStyle', LineStyles(idx));
        my_legends{idx} = sprintf('gNB%d', idx);
        hold on
    end
    %
    for idx = 1:n_gNBs
        corr_abs = abs(corrs{idx});
        peak_positions = find(corr_abs == max(corr_abs), 1);
        plot(samples(peak_positions), corr_abs(peak_positions), ...
            'Marker', Makers(idx),...
            'Color', cmap(idx+1,:), ...
            'LineWidth', 1 ...
            ); % Peaks
        my_legends{n_gNBs+idx} = '';
        hold on
        text(peak_positions, corr_abs(peak_positions), strcat('  gNB',num2str(idx)));
    end
    legend(my_legends);
    xlabel('Sample');
    ylabel('Absolute Value');

end
