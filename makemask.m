image_path1 = dir(fullfile('test_B','*.png'));
length(image_path1)
for i=1:length(image_path1)
    name1 = image_path1(i).name;
    shadow = im2double(imread([image_path1(i).folder '/' name1]));
    new = imgaussfilt(double(edge(shadow)),4)>0;
    % imshow([shadow new])
    imwrite(new, ['mask_new_edge/' name1])
end