folder_path = '../processed_dat';
files = dir(fullfile(folder_path, '*.dat'));

window_size = 256;
overlap = 200;
nfft = 512;

parfor i = 1:numel(files)
    if ~endsWith(files(i).name, '.dat')
        continue;
    end
    filename = fullfile(folder_path, files(i).name);
    if files(i).bytes == 0
        disp(['Skipping empty file: ', files(i).name]);
        continue;
    end
    fid = fopen(filename);
    waveform = fread(fid, inf, 'float');
    fclose(fid);

    waveform = waveform(waveform ~= 0);
    %waveform = log(waveform);
    
    [S, f, t] = spectrogram(waveform, window_size, overlap, nfft);

    h = figure('Visible', 'off');
    imagesc(t, f, 10*log10(abs(S)));
    axis xy;
    colormap(turbo);
    axis square;
    axis off;
    set(gca, 'Position', [0, 0, 1, 1])
    set(gcf, 'Units', 'pixels', 'Position', [100, 100, 100, 100]);
    set(gcf, 'PaperPosition', [0, 0, 1, 1]);
    [~, filename, ~] = fileparts(files(i).name);
    output_filename = fullfile('../dat_spectrogram', [filename, '.png']);
    saveas(h, output_filename, 'png');
    close(h);
end
