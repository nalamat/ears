[y, fs] = audioread('masker-flat-notch-noise-2k.wav');
[y2, fs2] = audioread('masker-mod-notch-tonal-2k.wav');

y = y(1:end-mod(length(y),fs));

rampDuration = 50e-3;
flatDuration = 100e-3;
gapDuration = 50e-3;

t = (1:rampDuration*fs)/fs;
ramp = sin(t*2*pi/rampDuration/4).^2;
flat = ones(1,flatDuration*fs);
gap = zeros(1,gapDuration*fs);
env = [ramp, flat, flip(ramp), gap];

% t = (1:length(env))/fs;
% plot(t, env);

yEnv = repmat(env', round(length(y)/length(env)), 1);
yMod = y .* yEnv;

t = (1:length(y))/fs;
msk = 35<t & t<=36;
% subplot(211); plot(t(msk)-35, yMod(msk));
% subplot(212); plot(t(msk)-35, y2(msk));

% audiowrite('mask-mod-notch-noise-2k.wav', yMod, fs);



% gen target
t = (1:fs)/fs;
y = sin(2*pi*2e3*t);
gap = zeros(size(env));
yCal = y .* [env env env env];
yGap2 = y .* [env gap env env];
yGap3 = y .* [env env gap env];

audiowrite('target-tonal-2k-cal.wav', yCal, fs);
audiowrite('target-tonal-2k-gap2.wav', yGap2, fs);
audiowrite('target-tonal-2k-gap3.wav', yGap3, fs);
