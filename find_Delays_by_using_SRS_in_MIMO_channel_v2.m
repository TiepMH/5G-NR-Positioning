clear all; clc;

n_TxAnt = 2;  % number of Tx antennas (1,2,4)
n_RxAnt = 4;  % number of Rx antennas
n_gNBs = 4; % number of gNBs = number of receivers

%% Carriers
% Create UE/carrier configuration
carrier = nrCarrierConfig;

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
figure
imagesc(abs(SRS_grid(:,:,1)));
title('Resource Grid for SRS (for the 1st antenna of UE)');
xlabel('OFDM symbol');
ylabel('Subcarrier');
axis xy

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

% For each gNB, there is a single nearby scatter.
scatter_pos{1} = 1e3*[1.9,    -5.2,    0.01];
scatter_pos{2} = 1e3*[2.9,    4.1,    0.01];
scatter_pos{3} = 1e3*[-6.2,    8.1,    0.01];
scatter_pos{4} = 1e3*[-8.8,    -1.5,    0.01];

% Distances from gNBs to UE
distances = zeros(1,n_gNBs);
for idx=1:n_gNBs
    distances(idx) = sqrt(sum(abs(gNB_pos{idx}-UE_pos).^2));
end

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
                      reshape(gNB_pos{idx},3,1), ...
                      reshape(UE_pos,3,1) ...
                      );
    PLs_dB{idx} = PL_dB;
end

%% Delays in the LOS path
LightSpeed = physconst('LightSpeed');
delays_LOS_in_seconds = zeros(1,n_gNBs);
delays_LOS_in_samples = zeros(1,n_gNBs);
delays_NLOS_in_seconds = zeros(1,n_gNBs);
delays_NLOS_in_samples = zeros(1,n_gNBs);
for idx = 1:n_gNBs
   distance = distances(idx);
   % Delays in the LOS paths
   delay_LOS_in_seconds = distance/LightSpeed; % Delay of the i-th gNB in seconds
   delays_LOS_in_seconds(idx) = delay_LOS_in_seconds; % Store this value
   delay_LOS_in_samples = round(delay_LOS_in_seconds*SampleRate); % Delay of the i-th gNB in samples
   delays_LOS_in_samples(idx) = delay_LOS_in_samples; % Store this value
   % Delays in the NLOS paths
   delay_NLOS_in_seconds = (sqrt(sum(abs(UE_pos - scatter_pos{idx}).^2)) ...
                            + sqrt(sum(abs(scatter_pos{idx} - gNB_pos{idx}).^2))) / LightSpeed;
   delays_NLOS_in_seconds(idx) = delay_NLOS_in_seconds; 
   delay_NLOS_in_samples = round(delay_NLOS_in_seconds*SampleRate); 
   delays_NLOS_in_samples(idx) = delay_NLOS_in_samples; 
end

%% MIMO Channel Configuration with both the LoS and NLOS components
channels = cell(1, n_gNBs);
Will_TxWaveForm_be_delayed_manually_later = true;
for idx=1:n_gNBs
    delay_LOS_in_seconds = delays_LOS_in_seconds(idx);
    delay_NLOS_in_seconds = delays_NLOS_in_seconds(idx);
    channels{idx} = ChannelObject(n_TxAnt, n_RxAnt, SampleRate, ...
                                  delay_LOS_in_seconds, delay_NLOS_in_seconds, ...
                                  Will_TxWaveForm_be_delayed_manually_later);
end


%% Signal-to-noise ratio (SNR in dB)
% Transmit power
Tx_power_dBm = 40; % Power (in dBm) delivered to all the Tx antennas for the fully-allocated grid 
Tx_power_dB = Tx_power_dBm - 30;
% Path loss in dB, between the UE and the 1-st gNB
PL_dB_gNB_1 = PLs_dB{1};
% SNR_dB per (resource element and receive antenna) is calculated as:
% NOTE: This SNR calculation has ALREADY included the impact of path loss
SNRdB_for_gNB1 = SNRdB_inclusive_of_PathLoss_per_RE_and_RxAntenna(Tx_power_dB, PL_dB_gNB_1, ...
                                                                  n_ResourceBlocks, Nfft, ...
                                                                  SampleRate);

%% On the Tx side (i.e., on the UE side):
% Perform OFDM modulation of SRS signals
txWaveform = nrOFDMModulate(carrier, SRS_grid);

size(txWaveform)

%% On the Rx side (i.e., on the 1-st gNB side):
% Manually delay the transmitted sequence
channel_1 = channels{1};
delay_LOS_in_samples = delays_LOS_in_samples(1);
MaximumChannelDelay = info(channel_1).MaximumChannelDelay;
txWaveform = [zeros(delay_LOS_in_samples, n_TxAnt); 
              txWaveform; ...
              zeros(MaximumChannelDelay - delay_LOS_in_samples, n_TxAnt)
              ];

% Transmission through a MIMO channel under the impact of multipath fading
rxWaveform_at_gNB1 = channel_1(txWaveform);

size(rxWaveform_at_gNB1) 

% SNR from SNRdB, wrt the 1-st gNB 
SNR_for_gNB1 = 10^(SNRdB_for_gNB1/10); % REMEMBER: this is per resource element & Rx antenna

% Normalize noise power by the IFFT size and by the number of Rx antennas
N0_normalized = 1/sqrt(n_RxAnt*Nfft*SNR_for_gNB1);

% Generate AWGN
noise = N0_normalized*complex(randn(size(rxWaveform_at_gNB1)), ...
                              randn(size(rxWaveform_at_gNB1)));

% Final received waveform at a certain gNB
rxWaveform_at_gNB1 = rxWaveform_at_gNB1 + noise;

%% Estimate time delays
% Let's consider the time delay estimation at the 1-st gNB

% CROSS-CORRELATION is used to estimate the time delay:
% In particular, we cross-correlate the input waveform and the reference waveform
% The input waveform is the 1st parameter
% The reference waveform is obtained by modulating the ref grid (the 3rd parameter)
% Matlab function: [offset,mag] = nrTimingEstimate(carrier,waveform,refGrid)
[offset,mag] = nrTimingEstimate(carrier, ...
                                rxWaveform_at_gNB1, ...
                                SRS_grid);

%% Display results
% size(txWaveform)
% 
% size(rxWaveform_at_gNB1)

disp(['Actual delay (in samples) between the UE and the 1-st gNB: ', num2str(delays_LOS_in_samples(1))]);

disp(['Estimated delay (in samples) between the UE and the 1-st gNB: ', num2str(offset)]);


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
                                 Will_TxWaveForm_be_delayed_manually_later)
    channel = nrTDLChannel;
    channel.DelayProfile = 'Custom'; % NOTE: We use this mode for Rician fading
    channel.FadingDistribution = 'Rician'; % Consider both LoS and NLOS components
    % First tap (LoS): Rician with K-factor 15 dB, path gain 0 dB, and path delay = delay_LOS_in_seconds
    % Second tap (NLoS-1): Rayleigh with average path gain −6 dB, and path delay = delay_NLOS_in_seconds 
    channel.KFactorFirstTap = 15; % in dB
    channel.AveragePathGains = [0, -6]; % [LoS gain, NLOS-1 gain] in dB
    if Will_TxWaveForm_be_delayed_manually_later
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