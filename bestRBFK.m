function [ model ] = bestRBFK( data, labels )
   model= BestRun(data, labels, '-t 2 -q');
end

