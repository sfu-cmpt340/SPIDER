folder_path = 'dat';
files = dir(fullfile(folder_path, '*.dat'));

window_size = 256;
overlap = 200;
nfft = 512;

parfor i = 1:numel(files)
    if ~endsWith(files(i).name, '.dat')
        continue;
    end
    filename = fullfile(folder_path, files(i).name);
    fid = fopen(filename, 'rb');
    waveform = fread(fid, inf, 'float');
    fclose(fid);

    waveform = waveform(waveform ~= 0);
    waveform = log(waveform);
    
    [S, f, t] = spectrogram(waveform, window_size, overlap, nfft);

    h = figure('Visible', 'off');
    imagesc(t, f, 10*log10(abs(S)));
    axis xy;
    colormap(turbo);
    axis square;
    axis off;
    set(gca, 'Position', [0, 0, 1, 1])
    set(gcf, 'Units', 'pixels', 'Position', [100, 100, 200, 200]); % Set figure size
    [~, filename, ~] = fileparts(files(i).name);
    output_filename = fullfile('spectrogram', [filename, '.png']);
    saveas(h, output_filename, 'png'); % Specify format
    close(h);
end
