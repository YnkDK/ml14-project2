function [pred] = softmaxPredict(theta, data)
    [~,pred] = max(data*theta,[],2);
end

