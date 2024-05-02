function saving_figs(fig_path)
    if ~isfolder(fig_path)
        error('The specified directory does not exist: %s', fig_path);
    end
    fig_handles = findobj('Type', 'figure');
    for i = 1:numel(fig_handles)
        fig = fig_handles(i);
        filename = sprintf('fig_%d.png', numel(fig_handles) - i + 1);
        fullpath = fullfile(fig_path, filename);
        if exist(fullpath, 'file')
            fprintf('Overwriting existing file: %s\n', fullpath);
        end
        set(fig, 'PaperSize', [8 6], 'PaperPosition', [0 0 8 6]);
        saveas(fig, fullpath);
    end
end
