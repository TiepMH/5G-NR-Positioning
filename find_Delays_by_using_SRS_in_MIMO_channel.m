clear all; clc;

n_TxAnt = 2;  % number of Tx antennas (1,2,4)
n_RxAnt = 4;  % number of Rx antennas
n_gNBs = 4; % number of gNBs = number of receivers

%% Carriers
% Create UE/carrier configuration
carrier = nrCarrierConfig;
SubcarrierSpacing = carrier.SubcarrierSpacing;

OFDM_Info = nrOFDMInfo(carrier);
Nfft = OFDM_Info.Nfft;
SampleRate = OFDM_Info.SampleRate;
SlotsPerFrame = OFDM_Info.SlotsPerFrame; % number of slots per frame
SymbolsPerSlot = OFDM_Info.SymbolsPerSlot; % number of symbols per slot

%% SRS Configuration
% Create an SRS-object
srs = nrSRSConfig;
srs.SymbolStart = 0;            % Starting OFDM symbol within a slot
srs.NumSRSSymbols = 4;          % Number of OFDM symbols allocated per slot (1,2,4)
srs.CSRS = 14;                  % Bandwidth configuration C_SRS (0...63). It controls the allocated bandwidth to the SRS
srs.KTC = 2;                    % Comb number (2,4). Frequency density in subcarriers
srs.SRSPeriod = [2 1];          % Periodicity and offset in slots. SRSPeriod(1) > SRSPeriod(2)

%% Generate PRS resource grid
% number of frames and number of slots
n_Frames = 1; % each frame is 10 ms long
total_Slots = n_Frames*SlotsPerFrame;

% Total number of resource blocks
n_ResourceBlocks = carrier.NSizeGrid;

% Total number of subcarriers
total_subcarriers = carrier.NSizeGrid * 12; 

% Total number of symbols in all frames
total_symbols = SymbolsPerSlot*total_Slots;

%% SRS resource grid
slots_containing_SRS_symbols = []; % to identify which slots contain SRS symbols

% Initialize the SRS grid of the UE by an empty object
SRS_grid = []; % Let's call it "the SRS grid"

for slot_k = 0:total_Slots-1
    
    % UPDATE the slot index for this carrier object
    carrier.NSlot = slot_k;
    
    % SRS symbols and indices at a specific slot
    SRS_symbols = nrSRS(carrier, srs);
    SRS_indices = nrSRSIndices(carrier, srs);
    
    % Create a resource grid spanning ONLY 1 slot in time domain
    grid_of_1_slot = nrResourceGrid(carrier,n_TxAnt); % size = (?, SymbolsPerSlot)

    % Map SRS symbols to the grid
    grid_of_1_slot(SRS_indices) = SRS_symbols;

    % Over time, "the SRS grid" will grow by 1 slot
    SRS_grid = [SRS_grid, grid_of_1_slot]; % expanded by 1 slot

    % % Save all transmitted OFDM grids and channel estimates for display purposes
    % thisSlot = slot_k*SymbolsPerSlot + (1:SymbolsPerSlot); % Symbols of the current slot
    % allTxGrid(:,thisSlot,:) = grid_of_1_slot;
    
    % Identify specific slots that contain PRS symbols
    if ~isempty(SRS_symbols)
        slots_containing_SRS_symbols = [slots_containing_SRS_symbols, slot_k];
    end % end if

end % looping-over-slots ends

%% Plot SRS resource grid, wrt 1st Tx antenna (i.e., wrt 1st antenna of UE)
SRS_grid_1st_antenna = SRS_grid(:, :, 1);
plot_SRS_grid(SRS_grid_1st_antenna)


%% Positions of gNBs in (x,y,z)-coordinate system
% UE position
UE_pos = [500 -20 1.8]; % 1.5m <= UE height <= 22.5m for 'UMa' path loss scenario
                       % see TR 38.901 Table 7.4.1-1
% Positions of gNBs
% height=25m for 'UMa' path loss scenario, see TR 38.901 Table 7.4.1-1 
gNBs_pos{1} = 1e3*[2,    -5,    0.0250]; 
gNBs_pos{2} = 1e3*[3,     4,    0.0250];
gNBs_pos{3} = 1e3*[-6,    8,    0.0250];
gNBs_pos{4} = 1e3*[-7,   -3,    0.0250];

% For each gNB, there is a single nearby scatter.
n_scatters = 1;
azRange = -180:180;
elRange = -90:90;
for idx=1:n_gNBs
    randAzOrder = randperm(length(azRange));
    randElOrder = randperm(length(elRange));
    azAngInSph = azRange(randAzOrder(1:n_scatters));
    elAngInSph = elRange(randElOrder(1:n_scatters));
    radius_temp = 1e3;            % radius - a temporary variable
    [x_temp,y_temp,z_temp] = sph2cart(deg2rad(azAngInSph),deg2rad(elAngInSph),radius_temp);
    % Find all scatters associated with gNB_i 
    scatters_pos{idx} = [x_temp,y_temp,z_temp] + (0.5*UE_pos + 0.5*gNBs_pos{idx}); %#ok
end

% Distances from UE to gNBs
distances = zeros(1,n_gNBs);
for idx=1:n_gNBs
    distances(idx) = sqrt(sum(abs(gNBs_pos{idx}-UE_pos).^2));
end

%% Plot the scene
plot_system(UE_pos, gNBs_pos, scatters_pos)
view(2)

%% Path Loss Configuration
PathLoss = nrPathLossConfig;
PathLoss.Scenario = 'Uma';
fc = 3e9;    % Carrier frequency (Hz)
PLs_dB = cell(1, n_gNBs);
for idx = 1:n_gNBs
    % Calculate path loss for each gNB and UE pair
    line_of_sight = true; % There is the line of sight (LOS) component
    PL_dB = nrPathLoss(PathLoss, fc, ...
                      line_of_sight, ...
                      reshape(gNBs_pos{idx},3,1), ...
                      reshape(UE_pos,3,1) ...
                      );
    PLs_dB{idx} = PL_dB;
end

%% Delays wrt the LOS and NLOS paths
LightSpeed = physconst('LightSpeed');
delays_LOS_in_seconds = zeros(1,n_gNBs);
delays_LOS_in_samples = zeros(1,n_gNBs);
delays_NLOS_in_seconds = zeros(1,n_gNBs);
delays_NLOS_in_samples = zeros(1,n_gNBs);
for idx = 1:n_gNBs
   distance = distances(idx);
   % Delays wrt the LOS paths
   delay_LOS_in_seconds = distance/LightSpeed; % Delay of the i-th gNB in seconds
   delays_LOS_in_seconds(idx) = delay_LOS_in_seconds; % Store this value
   delay_LOS_in_samples = round(delay_LOS_in_seconds*SampleRate); % Delay of the i-th gNB in samples
   delays_LOS_in_samples(idx) = delay_LOS_in_samples; % Store this value
   % Delays wrt the NLOS paths
   delay_NLOS_in_seconds = (sqrt(sum(abs(UE_pos - scatters_pos{idx}).^2)) ...
                            + sqrt(sum(abs(scatters_pos{idx} - gNBs_pos{idx}).^2))) / LightSpeed;
   delays_NLOS_in_seconds(idx) = delay_NLOS_in_seconds; 
   delay_NLOS_in_samples = round(delay_NLOS_in_seconds*SampleRate); 
   delays_NLOS_in_samples(idx) = delay_NLOS_in_samples; 
end

%% MIMO Channel Configuration with both the LoS and NLOS components
channels = cell(1, n_gNBs);
Do_we_delay_TxWaveForm_manually = true;
for idx=1:n_gNBs
    delay_LOS_in_seconds = delays_LOS_in_seconds(idx);
    delay_NLOS_in_seconds = delays_NLOS_in_seconds(idx);
    channels{idx} = ChannelObject(n_TxAnt, n_RxAnt, SampleRate, ...
                                  delay_LOS_in_seconds, delay_NLOS_in_seconds, ...
                                  Do_we_delay_TxWaveForm_manually);
end


%% Signal-to-noise ratio (SNR in dB)
% Transmit power
Tx_power_dBm = 40; % Power (in dBm) delivered to all the Tx antennas for the fully-allocated grid 
Tx_power_dB = Tx_power_dBm - 30;
SNRdB_for_all_gNBs = cell(1, n_gNBs);
for idx=1:n_gNBs
    % Path loss in dB, between the UE and the i-th gNB
    PL_dB_gNB_i = PLs_dB{idx};
    % SNR_dB per (resource element and receive antenna) is calculated as:
    % NOTE: This SNR calculation has ALREADY included the impact of path loss
    SNRdB_for_gNB_i = SNRdB_inclusive_of_PathLoss_per_RE_and_RxAntenna(Tx_power_dB, PL_dB_gNB_i, ...
                                                                       n_ResourceBlocks, Nfft, ...
                                                                       SampleRate);
    % Store values
    SNRdB_for_all_gNBs{idx} = SNRdB_for_gNB_i;
end

%% On the Tx side (i.e., on the UE side):
% Perform OFDM modulation of SRS signals
txWaveform = nrOFDMModulate(carrier, SRS_grid);

size(txWaveform)

%% At all gNBs:
rxWaveforms_at_all_gNBs = cell(1, n_gNBs);
for idx=1:n_gNBs
    % Manually delay the transmitted sequence
    channel_i = channels{idx};
    delay_LOS_in_samples = delays_LOS_in_samples(idx);
    MaximumChannelDelay = info(channel_i).MaximumChannelDelay;
    txWaveform_that_is_delayed_when_reaching_gNB_i = [zeros(delay_LOS_in_samples, n_TxAnt); 
                                                      txWaveform; ...
                                                      zeros(MaximumChannelDelay - delay_LOS_in_samples, n_TxAnt)
                                                      ];
    
    % Transmission through a MIMO channel under the impact of multipath fading
    rxWaveform_at_gNB_i = channel_i(txWaveform_that_is_delayed_when_reaching_gNB_i);
    
    % SNRdB wrt the i-th gNB 
    SNRdB_for_gNB_i = SNRdB_for_all_gNBs{idx};
    % SNR from SNRdB
    SNR_for_gNB_i = 10^(SNRdB_for_gNB_i/10); % REMEMBER: this is per resource element & Rx antenna
    
    % Normalize noise power by the IFFT size and by the number of Rx antennas
    N0_normalized = 1/sqrt(n_RxAnt*Nfft*SNR_for_gNB_i);
    
    % Generate AWGN
    noise_at_gNB_i = N0_normalized*complex(randn(size(rxWaveform_at_gNB_i)), ...
                                           randn(size(rxWaveform_at_gNB_i)));
    
    % Final received waveform at a certain gNB
    rxWaveforms_at_all_gNBs{idx} = rxWaveform_at_gNB_i + noise_at_gNB_i;
end

%% Estimate time delays
% Let's consider the time delay estimation at the 1-st gNB

% CROSS-CORRELATION is used to estimate the time delay:
% In particular, we cross-correlate the input waveform and the reference waveform
% The input waveform is the 1st parameter
% The reference waveform is obtained by modulating the ref grid (the 3rd parameter)
% Matlab function: [offset,mag] = nrTimingEstimate(carrier,waveform,refGrid)
offsets = cell(1, n_gNBs);
corrs = cell(1, n_gNBs);
for idx=1:n_gNBs
    % Offset (time delay in samples) and magnitude (amplitude of correlation), wrt the i-th gNB
    [offset_i,mag_i] = nrTimingEstimate(carrier, ...
                                        rxWaveforms_at_all_gNBs{idx}, ...
                                        SRS_grid);

    % Ignore noisy side lobe peaks
    corrs{idx} = mag_i(1:(Nfft*SubcarrierSpacing/15));

    % Store values
    offsets{idx} = offset_i;
end

% Plot PRS correlation results
plot_SRS_correlations(corrs, SampleRate); 

%% Display results
disp(['Actual delays (in samples) between the UE and all gNBs: ', num2str(delays_LOS_in_samples)]);

disp(['Estimated delays (in samples) between the UE and all gNBs: ', num2str(cell2mat(offsets))]);


%% Functions
function SNR_dB = SNRdB_inclusive_of_PathLoss_per_RE_and_RxAntenna(Tx_power_dB, PL_dB, ...
                                                                   n_ResourceBlocks, Nfft, ...
                                                                   SampleRate)
    % Reference: https://www.mathworks.com/help/5g/ug/include-path-loss-in-nr-link-level-simulations.html
    % Transmit power
    Tx_power_in_Watts = 10^(Tx_power_dB/10);
    % Path loss
    PL = 10^(PL_dB/10);
    % Average receive power (including path loss) per resource element and per Rx antenna
    S_per = (Tx_power_in_Watts/PL) * (Nfft^2) / (12*n_ResourceBlocks);
    % ACTUAL noise amplitude N0 per Rx antenna
    kBoltz = physconst('Boltzmann');
    Temperature = 350; % Kelvin
    N0_actual = sqrt(kBoltz*SampleRate*Temperature/2); % ACTUAL noise amplitude
    % Average noise per resource element and per Rx antenna
    N_per = 2*(N0_actual^2)*Nfft;
    % SNR per resource element and per Rx antenna
    SNR = S_per/N_per;
    % Convert SNR to SNRdB
    SNR_dB = 10*log10(SNR);
end

function channel = ChannelObject(n_TxAnt, n_RxAnt, SampleRate, ...
                                 delay_LOS_in_seconds, delay_NLOS_in_seconds, ...
                                 Do_we_delay_TxWaveForm_manually)
    channel = nrTDLChannel;
    channel.DelayProfile = 'Custom'; % NOTE: We use this mode for Rician fading
    channel.FadingDistribution = 'Rician'; % Consider both LoS and NLOS components
    % First tap (LoS): Rician with K-factor 15 dB, path gain 0 dB, and path delay = delay_LOS_in_seconds
    % Second tap (NLoS-1): Rayleigh with average path gain −6 dB, and path delay = delay_NLOS_in_seconds 
    channel.KFactorFirstTap = 15; % in dB
    channel.AveragePathGains = [0, -6]; % [LoS gain, NLOS-1 gain] in dB
    if Do_we_delay_TxWaveForm_manually
        % Relative time delays
        channel.PathDelays = [0, delay_NLOS_in_seconds - delay_LOS_in_seconds]; % [for LoS, for NLOS-1]
        % NOTE: 
        % In this case, Tx waveform will be delayed manually before applying the channel() function.
        % For example, txWaveform = [11, 22, 33], delay_LOS = 2 samples
        % We have to modify txWaveform into txWaveform = [0, 0, 11, 22, 33]
        % before using the command: txWaveform = channel(txWaveform);
    else
        % Exact time delays
        channel.PathDelays = [delay_LOS_in_seconds, delay_NLOS_in_seconds]; % [for LoS, for NLOS-1]
        % NOTE:
        % In this case, we do not need to delay the transmitted sequence manually.
        % However, when we write: txWaveform = channel(txWaveform);
        % the output and the input ALWAYS have the same size ==>> the same length.
        % Having the same length is a drawback!!!
        % When there is no difference in length, the impact of time delay is NOT visually apparent, ...
        % ... making the code harder to interpret.
        % Additionally, extending the length of txWaveform in the main code will be incorrect.
    end
    % Consider MIMO channel
    channel.NumTransmitAntennas = n_TxAnt; % number of Tx antennas
    channel.NumReceiveAntennas = n_RxAnt; % number of Rx antennas
    % Set channel sample rate
    channel.SampleRate = SampleRate;
    %
    % channel.TransmissionDirection = 'Uplink'; % no need
    MaximumChannelDelay = info(channel).MaximumChannelDelay;
end

function plot_system(UE_pos, gNBs_pos, scatters_pos)
n_gNBs = numel(gNBs_pos);
figure
fig_UE = plot3(UE_pos(1),UE_pos(2),UE_pos(3),'k^',LineWidth=3);
hold on;
for idx=1:n_gNBs
    fig_gNBs(idx) = scatter3(gNBs_pos{idx}(1),gNBs_pos{idx}(2),gNBs_pos{idx}(3),'rs',LineWidth=3); %#ok
    hold on;
    fig_scatters(idx) = plot3(scatters_pos{idx}(1),scatters_pos{idx}(2),scatters_pos{idx}(3),'kx',LineWidth=2); %#ok
    hold on;
    fig_LOS_paths(idx) = plot3([UE_pos(1) gNBs_pos{idx}(1)], ...
                               [UE_pos(2) gNBs_pos{idx}(2)], ...
                               [UE_pos(3) gNBs_pos{idx}(3)], ...
                               'k'); %#ok
    hold on;
    fig_NLOS_paths(idx) = plot3([UE_pos(1) scatters_pos{idx}(1) gNBs_pos{idx}(1) scatters_pos{idx}(1)], ...
                                [UE_pos(2) scatters_pos{idx}(2) gNBs_pos{idx}(2) scatters_pos{idx}(2)], ...
                                [UE_pos(3) scatters_pos{idx}(3) gNBs_pos{idx}(3) scatters_pos{idx}(3)], ...
                                'k--'); %#ok
end
xlabel('x-axis (m)')
ylabel('y-axis (m)')
zlabel('z-axis (m)')
legend([fig_UE, fig_gNBs(1), fig_scatters(1), fig_LOS_paths(1), fig_NLOS_paths(1)], ...
       {'UE','gNB','Scatter','LOS path','NLOS path'}, ...
       'Location', 'bestoutside');
end

function plot_SRS_grid(SRS_grid)
    num_colors = 1 + 1; % background + color for SRS
    my_legends = [{' '}, 'SRS from UE']; % background + SRS

    % uisetcolor is a color picker
    % define the colormap by a 3-column matrix of RGB
    cmap = jet(num_colors);
    cmap(1,:) = [1 1 1]; % white background 
    cmap(2,:) = [0 0 1]; % blue color for SRS from UE 1
    % cmap(3,:) = [0 1 0]; % green color for SRS from UE 2
    % cmap(4,:) = [1 0 1]; % magenta color for SRS from UE 3
    % cmap(5,:) = [0 1 1]; % cyan color for SRS from UE 4
    % cmap(6,:) = [1 0.4 0.2]; % orange color for SRS from UE 5

    % SRS grid
    grid_of_abs_values_for_SRS = abs(SRS_grid);
    grid_of_abs_values_for_SRS(grid_of_abs_values_for_SRS ~= 0) = 1; % background=0
    
    %
    figure
    imagesc(grid_of_abs_values_for_SRS);
    colormap(cmap(1:2,:));
    hold on
    %
    L = line(ones(num_colors),ones(num_colors), 'LineWidth',2); 
    set(L,{'color'},mat2cell(cmap(1:end,:),ones(1,num_colors),3)); 
    legend([L(2)], my_legends(2)); % L(1) is background. L(2) is SRS
    %
    title('Resource Grid for SRS (for the 1st antenna of UE)');
    xlabel('OFDM symbol');
    ylabel('Subcarrier');
    axis xy
end

function plot_SRS_correlations(corrs, SampleRate)
    n_gNBs = numel(corrs);
    num_colors = 1 + n_gNBs; % background + colors for gNBs

    % uisetcolor is a color picker
    % define the colormap by a 3-column matrix of RGB
    cmap = jet(num_colors);
    cmap(1,:) = [1 1 1]; % white background for prsGrid_many_gNBs
    cmap(2,:) = [0 0 1]; % blue color for PRS from gNB 1
    cmap(3,:) = [0 1 0]; % green color for PRS from gNB 2
    cmap(4,:) = [1 0 1]; % magenta color for PRS from gNB 3
    % cmap(5,:) = [0 1 1]; % cyan color for PRS from gNB 4
    % cmap(6,:) = [1 0.4 0.2]; % orange color for PRS from gNB 5

    % Line widths and 
    LineWidths = [1, 1, 1, 1, 1];
    LineStyles = ['-', '-', '-', '-', '-'];
    Makers = ['o', "square", '*', "diamond", '>'];

    figure
    % Plot correlation for gNBs
    samples = (0:length(corrs{1}) - 1);
    time_in_seconds = (0:length(corrs{1}) - 1)/SampleRate;
    my_legends = cell(1,2*n_gNBs);
    for idx = 1:n_gNBs
        subplot(n_gNBs,1,idx);
        plot(samples, abs(corrs{idx}), ...
            'Color', cmap(idx+1,:), ...
            'LineWidth', LineWidths(idx), ...
            'LineStyle', LineStyles(idx));
        hold on
        %
        corr_abs = abs(corrs{idx});
        peak_positions = find(corr_abs == max(corr_abs), 1);
        plot(samples(peak_positions), corr_abs(peak_positions), ...
            'Marker', Makers(idx),...
            'Color', cmap(idx+1,:), ...
            'LineWidth', 1 ...
            ); % Peaks
        %
        text(peak_positions, corr_abs(peak_positions), strcat('  gNB',num2str(idx)));
        my_legends = sprintf('gNB%d', idx);
        legend(my_legends);
        xlabel('Sample');
        ylabel('Abs. Value');
    end
    %
end
