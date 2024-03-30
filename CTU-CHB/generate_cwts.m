folder_path = 'processed_dat';
files = dir(fullfile(folder_path, '*.dat'));

scales = 1:256;
waveletName = 'morse';

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
    
    [cfs, frequencies] = cwt(waveform);

    h = figure('Visible', 'off');
    imagesc(1:numel(waveform), frequencies, abs(cfs));
    colormap(turbo);
    axis xy;
    axis square;
    axis off;
    set(gca, 'Position', [0, 0, 1, 1])
    set(gcf, 'Units', 'pixels', 'Position', [100, 100, 100, 100]);
    set(gcf, 'PaperPosition', [0, 0, 1, 1]);
    [~, filename, ~] = fileparts(files(i).name);
    output_filename = fullfile('cwt', [filename, '.png']);
    saveas(h, output_filename, 'png');
    close(h);
end
