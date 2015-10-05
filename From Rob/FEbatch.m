gcp
batches=10;
ntrees=100;
[n,m]=size(xtest2);
yhat=zeros(n,9);
acc=zeros(batches,1);
tic;
for j=1:batches
    for k=1:ntrees

        ff=fitensemble(xtr2,ytr2,'Bag',1,'Tree','type','classification');
        % 82.5% accuracy
        
        %ff=TreeBagger(1,xos,yos,'Method','classification','NVarToSample',80);

        yh=predict(ff,xtest2);
        for i=1:n
            yhat(i,yh(i))=yhat(i,yh(i))+1;
        end
    end
    [junk,yh]=max(yhat,[],2);
    acc(j)=sum(yh==ytest2)/length(ytest2);
    disp([j,acc(j)]);
end

toc
beep
beep
beep