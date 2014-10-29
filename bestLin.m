function [ model ] = bestLin( data, labels )
   model= BestRun(data, labels, '-t 0 -q -c %f -g %f -v %d ');
end

