fs = 100e3;
fc = 1e3;

% generate the flat stimulus
flat_len = 1;
flat_len = flat_len-rem(flat_len,1/fc); % make it repeatable
flat_len = flat_len - 1/fs;

t = 0:1/fs:flat_len;
y = sin(2*pi*fc*t);
% plot(t, y);
% xlim([0 flat_len]);
% grid on;
% pause;

% audiowrite('T01_000.wav', y, fs);
% a = audioplayer([y y y y y y y y y y y y y y], fs);
% play(a);
% return;

% generate the ramp
ramp_len = 50e-3;
ramp_len = ramp_len-rem(ramp_len,1/fc); % make it repeatable

t2 = 0:1/fs:ramp_len;
ramp = sin(2*pi*1/ramp_len/4*t2).^2;

y(1:length(ramp)) = y(1:length(ramp)) .* ramp;
y(end-length(ramp)+1:end) = y(end-length(ramp)+1:end) .* ramp(end:-1:1);
plot(t, y);
xlim([0 flat_len]);
grid on;

audiowrite('T01_ramped.wav', y, fs);
