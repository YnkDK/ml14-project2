function [ model ] = BestPoly( data, labels )
   model= BestRun(data, labels, '-t 1 -q');
end

