function [ model ] = BestPoly( data, labels )
   model= BestRun(data, labels, '-t 1 -q -c %f -g %f -v %d ');
end

