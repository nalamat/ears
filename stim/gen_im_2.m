% gap between consecutive set of puleses for continious nogo: random 5 to 7 secs
% start continous background nogo from 10 khz and rove down to 2 khz
% introduce actual nogo at 2 khz

clear
close all
clc
Fs = 44.1e3;
t = 0:1/Fs:300e-3;%300 ms long tones

x = sin(2*pi*1e3*t);%1kHz tone -- need to add cos^2 ramp or some such to avoid speaker distortion
sil = 0.*[0:1/Fs:200e-3];%200 ms inter-tone-interval
b = [x sil];%beep

Go = [sil b b 0.*b 0.*b b];
NoGo = [sil b 0.*b 0.*b b b];

pgo = audioplayer(Go,Fs);
pnogo = audioplayer(NoGo,Fs);

playblocking(pgo)
pause
playblocking(pnogo)
