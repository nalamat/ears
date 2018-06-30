clear
close all
clc
spth = '.\'; % path where stimuli are saved
rms = inline('sqrt(mean(x.^2))','x');
t.Fs = 100e3; % sampling frequency in Hz
t.Duration = 30; % total stimulus duration in s [That's exactly 8 cycles @ 16 Hz]
t.Ramp = 10e-3; % ramp duration
t.TargetMaskerDelay = 156.3e-3; %  time delay between onset of the masker and onset of the target

fc = 1e3;
T.fc = fc;

t.EnvelopeAMfreq = 10; % envelope amplitude modulation frequency
t.EnvelopeAMhalfcycle = 0.5*1/t.EnvelopeAMfreq;
t.EnvelopeAMdepth = 100; % envelope depth in %
t.EnvelopeRamp = 10e-3; % envelope ramp in s

M.how_many_noise_frequency_components = 500;

t.t = 0 : 1/t.Fs : (t.Duration-1/t.Fs); % time vector
t.ramp = sin(0:.5*pi/round(t.Ramp*t.Fs):pi/2).^2; % ramp time vector
t.envelopehalfcycle = ones(round(t.EnvelopeAMhalfcycle*t.Fs),1);
t.enveloperamp = sin(0:.5*pi/round(t.EnvelopeRamp*t.Fs):pi/2).^2;
t.envelopewindow = 0.*t.envelopehalfcycle + 1;
t.envelopewindow(1:length(t.enveloperamp)) = t.enveloperamp;
t.envelopewindow(end-length(t.enveloperamp)+1:end) = fliplr(t.enveloperamp);
t.envelopewindow = [t.envelopewindow' 0.*t.envelopewindow(1:end-1)']; % drop one sample of the zero portion to fit the entire signal into 500 ms
t.envelopewindow = repmat(t.envelopewindow,1,round(t.Duration/t.t(length(t.envelopewindow)))); % time vector of the envelope of the masker

t.envelopePhase0 = [ t.envelopewindow zeros(round(t.EnvelopeAMhalfcycle*t.Fs/2),1)'];
t.envelopePhase0  = t.envelopePhase0(1:length(t.t));
t.envelopePhasePi = [zeros(round(t.EnvelopeAMhalfcycle*t.Fs),1)' t.envelopewindow];
t.envelopePhasePi  = t.envelopePhasePi(1:length(t.t));
t.envelopeUnmodulated = [t.enveloperamp ones(length(t.t)-2*length(t.enveloperamp),1)' fliplr(t.enveloperamp )];


M.fNoise = T.fc*2^(-1/3) : 1/t.t(end) : T.fc*2^(1/3);

for which_noise_token = 1
	disp(['Target: ',num2str(fc),', Noise: ',num2str(which_noise_token)])

	M.which_comp = randperm(length(M.fNoise));
	M.which_comp = M.which_comp(1:M.how_many_noise_frequency_components);
	M.comp_phase = rand(size(M.which_comp))*pi;
	M.M = 0.*t.t;
	for which_comp = 1 : length(M.which_comp)
		M.M = M.M + sin(2*pi*M.fNoise(M.which_comp(which_comp))*t.t + M.comp_phase(which_comp));
	end

	t.MModPi = M.M.* t.envelopePhasePi;
	t.MMod0 = M.M.* t.envelopePhase0;

	t.MMod0 = t.MMod0/rms(t.MMod0);
	t.MModPi = t.MModPi/rms(t.MModPi);

	masker = t.MMod0/10;

	% 'M' -F -E -Fc
	% F: On-Target = 0; Flanker = 1; Relative Frequency
	% E: Unmod = 0; Mod0 = 1; ModPi = 2; Envelope characteristics
	% Fc: Low = 0; Mid = 1; High = 2; Center Frequency in Hz

	plot((1:size(masker,2))/t.Fs, masker);
	audiowrite('Supermasker.wav', masker, t.Fs, 'bitspersample', 16); % on target mod0
end
