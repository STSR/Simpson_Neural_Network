%% Classification of Simpson characters code

%% Load Images form simpsons_dataset folder and classify by subfolder
simpsonDatabase = imageSet('simpsons_dataset','recursive');

%% Split Database into 80% Training and 20% Testtestng Sets
[training,test] = partition(simpsonDatabase,[0.8 0.2]);

%% Extract HOG Features for training set 
trainingFeatures = zeros(size(training,2)*training(1).Count,4680);
featureCount = 1;
for i=1:size(training,2)
    for j = 1:training(i).Count
        %Creates huge matrix of 
        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),j));
        % List that shows which features correspond to which character
        trainingLabel{featureCount} = training(i).Description;    
        featureCount = featureCount + 1;
    end
    personIndex{i} = training(i).Description;
end



%% Using fitcecoc function to learn how to dfferentiate between characters
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);

%% Test test Characters from Test Set
figure;
figureNum = 1;
correct = 0;
error = 0;
for Character=1:size(test,2)
    for j = 1:test(Character).Count
        queryImage = read(test(Character),j);
        queryFeatures = extractHOGFeatures(queryImage);
        personLabel = predict(faceClassifier,queryFeatures);
        % Map back to training set to find identity
        booleanIndex = strcmp(personLabel, personIndex);
        integerIndex = find(booleanIndex);
        
        %% Make plot of ansers
        subplot(2,2,figureNum);
        imshow(imresize(queryImage,3));
        title('Guessed Class');
        subplot(2,2,figureNum+1);
        imshow(imresize(read(training(integerIndex),1),3));
        title('Actual Class');

        if(figureNum == figureNum+1){
            correct = correct + 1;

        }else{
            error = error + 1;
        }

        figureNum = figureNum+2;   
    end
    figure;
    figureNum = 1;

end

fprintf('%i\n', correct);
fprintf('%i\n', error);


