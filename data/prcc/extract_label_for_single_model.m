clear

train_idx=load('./exp/train_id.mat');
% test_idx=load('./exp/test_id.mat');
val_idx=load('./exp/val_id.mat');

train_idall=[train_idx.id val_idx.id];

% train_idall=train_idx.id;


fid=fopen('train_val_softmax_single_model.txt','wt');
number1=0;
number2=0;
%cam1
str1_all=[];
for i=1:size(train_idall,2)
    i
    if train_idall(i)<10
        str1=strcat('000',num2str(train_idall(i)));
    elseif train_idall(i)<100
        str1=strcat('00',num2str(train_idall(i)));
    else
        str1=strcat('0',num2str(train_idall(i)));
    end
%     str1_all=[str1_all str1];
%     str1
    id_select=train_idall(i);
%     for j=[1:2,4:5]
    for j=1:6
%         strcat('./sysu-mm01-data/cam',num2str(j),'/',str1,'/*.jpg')
        dir1=dir(strcat('./sysu-mm01-data/cam',num2str(j),'/',str1,'/*.jpg'));
%         number2=number2+size(dir1,1);
        for z=1:size(dir1,1)
            pic_name=strcat('sysu-mm01-data/cam',num2str(j),'/',str1,'/',dir1(z).name);
%             fprintf(fid,'%s %d %d\n',pic_name,i-1,j-1);
            fprintf(fid,'%s %d\n',pic_name,i-1);
%             number1=number1+1;
        end
    end
end