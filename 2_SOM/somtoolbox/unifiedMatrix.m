function [] = unifiedMatrix(m)
%input is SOM struct sm
figure;
colorMap = colormap('gray');
colormapigray = ones(size(colorMap, 1),size(colorMap, 2)) - colorMap;
colormap(colormapigray);
Um = som_umat(m);
som_cplane('hexaU', m.topol.msize, Um(:));%draw umatrix
%h.Position = [2507 189 756 696];
end