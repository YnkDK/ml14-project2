function [ model ] = bestLin( data, labels )
   model= BestRun(data, labels, '-t 0 -q ');
end

