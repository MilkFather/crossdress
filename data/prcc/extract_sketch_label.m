clear

fidin=fopen('train_prcc_sketch.txt','w');


folder_all=dir('./sketch/train/');



for i=3:size(folder_all,1)
    folder_name=folder_all(i).name;
    files=dir(strcat('sketch/train/',folder_name,'/*.jpg'));
    for j=1:size(files,1)
        pic_name=strcat('prcc/sketch/train/',folder_name,'/',files(j).name);
        fprintf(fidin,'%s %d\n',pic_name,i-3);
    end
end

fclose(fidin);



fidin=fopen('test_prcc_sketch_A.txt','w');


folder_all=dir('./sketch/test/A/');



for i=3:size(folder_all,1)
    folder_name=folder_all(i).name;
    files=dir(strcat('sketch/test/A/',folder_name,'/*.jpg'));
    for j=1:size(files,1)
        pic_name=strcat('prcc/sketch/test/A/',folder_name,'/',files(j).name);
        fprintf(fidin,'%s %d\n',pic_name,i-3);
    end
end

fclose(fidin);

fidin=fopen('test_prcc_sketch_B.txt','w');


folder_all=dir('./sketch/test/B/');



for i=3:size(folder_all,1)
    folder_name=folder_all(i).name;
    files=dir(strcat('sketch/test/B/',folder_name,'/*.jpg'));
    for j=1:size(files,1)
        pic_name=strcat('prcc/sketch/test/B/',folder_name,'/',files(j).name);
        fprintf(fidin,'%s %d\n',pic_name,i-3);
    end
end

fclose(fidin);

fidin=fopen('test_prcc_sketch_C.txt','w');


folder_all=dir('./sketch/test/C/');



for i=3:size(folder_all,1)
    folder_name=folder_all(i).name;
    files=dir(strcat('sketch/test/C/',folder_name,'/*.jpg'));
    for j=1:size(files,1)
        pic_name=strcat('prcc/sketch/test/C/',folder_name,'/',files(j).name);
        fprintf(fidin,'%s %d\n',pic_name,i-3);
    end
end

fclose(fidin);