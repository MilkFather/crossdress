clear

fidin=fopen('train_prcc_color.txt','w');


folder_all=dir('./rgb/train/');



for i=3:size(folder_all,1)
    folder_name=folder_all(i).name;
    files=dir(strcat('rgb/train/',folder_name,'/*.jpg'));
    for j=1:size(files,1)
        pic_name=strcat('prcc/rgb/train/',folder_name,'/',files(j).name);
        fprintf(fidin,'%s %d\n',pic_name,i-3);
    end
end

fclose(fidin);



fidin=fopen('test_prcc_color_A.txt','w');


folder_all=dir('./rgb/test/A/');



for i=3:size(folder_all,1)
    folder_name=folder_all(i).name;
    files=dir(strcat('rgb/test/A/',folder_name,'/*.jpg'));
    for j=1:size(files,1)
        pic_name=strcat('prcc/rgb/test/A/',folder_name,'/',files(j).name);
        fprintf(fidin,'%s %d\n',pic_name,i-3);
    end
end

fclose(fidin);

fidin=fopen('test_prcc_color_B.txt','w');


folder_all=dir('./rgb/test/B/');



for i=3:size(folder_all,1)
    folder_name=folder_all(i).name;
    files=dir(strcat('rgb/test/B/',folder_name,'/*.jpg'));
    for j=1:size(files,1)
        pic_name=strcat('prcc/rgb/test/B/',folder_name,'/',files(j).name);
        fprintf(fidin,'%s %d\n',pic_name,i-3);
    end
end

fclose(fidin);

fidin=fopen('test_prcc_color_C.txt','w');


folder_all=dir('./rgb/test/C/');



for i=3:size(folder_all,1)
    folder_name=folder_all(i).name;
    files=dir(strcat('rgb/test/C/',folder_name,'/*.jpg'));
    for j=1:size(files,1)
        pic_name=strcat('prcc/rgb/test/C/',folder_name,'/',files(j).name);
        fprintf(fidin,'%s %d\n',pic_name,i-3);
    end
end

fclose(fidin);