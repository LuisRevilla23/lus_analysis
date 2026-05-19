% These two sections of code are to process the LUS scans of the COVID-19
% phantom from the manuscript https://doi.org/10.1016/j.ultras.2024.107251
% please cite this paper if you use this dataset

% James R McLaughlan
% Jan 2024
% j.r.mclaughlan@leeds.ac.uk

%% To convert all video files to their individial frames

vidNumber  = ls('*.webm');
workingDir = ''; % put in filepath to video clips
saveDir    = ''; % put in filepath to where you want the frames to be saved

for j=1:length(vidNumber)
    
    dp    = find(vidNumber(j,:)== '.');
    fName = vidNumber(j,1:dp-1);disp(['Processing ' fName '-' num2str(j) ' of ' num2str(length(vidNumber))]);
    v     = VideoReader([fName '.webm']);
    ii    = 1;
    %mkdir(workingDir,fName);
    
    while hasFrame(v)
       img = readFrame(v);
       filename = [fName '-' sprintf('F%02d',ii) '.jpg'];
       %fullname = fullfile(workingDir,fName,filename);
       fullname = fullfile(saveDir,filename);
       imwrite(img,fullname)    % Write out to a JPEG file (img1.jpg, img2.jpg, etc.)
       ii = ii+1;
    end
    
end

%% To randomise images between differet 'users' who are going to label the images. 

workingDir = ''; % put in filepath to where all frames are saved
saveDir    = ''; % put in filepath to where you would want subgroups of images to be saved to
uName      = ['User1';'User2';'User3';'User4']; % list number of different users who will label the images
nF         = 400; % total number of frames split between users. Please note that it should avoid duplications.
picNumber  = ls('*.jpg');
p = randperm(length(picNumber),nF);

    for i=1:nF
        
        if i<=100
            j=1;
        elseif i>=101 && i<=200
            j=2;
        elseif i>=201 && i<=300
            j=3;
        else
            j=4;
        end
        
        status = movefile(picNumber(p(i),:),fullfile(saveDir,uName(j,:)));
        
        if status == 0
            disp(['Did not copy at ' num2str(i)]);
            break
        end
    end
