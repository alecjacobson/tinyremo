for pass = 1:3
  switch pass
  case 1
    tit = 'Log-log plot of \nabla (llt) runtime';
    meths = {'eigen','tinyad','tinyremo'};
    D = [
      2 1.70946e-06 2.19345e-07 1.03951e-06
4 5.81026e-06 1.27077e-06 2.89917e-06
8 3.32093e-05 2.02608e-05 1.59311e-05
16 0.000130291 0.000259588 5.12099e-05
32 0.000580311 0.00651231 0.000296741
64 0.00385405 0.25113 0.0016996
128 0.011595 nan 0.00254929
      ];
  case 2
    tit = 'Log-log plot of \nabla^2 (llt) runtime';
    meths = {'tinyad','tinyremo'};
    D = [
2 8.10623e-08 1.82152e-06
4 1.15871e-06 6.99997e-06
8 2.01201e-05 5.21398e-05
16 0.000257361 0.000242279
32 0.00669836 0.00194119
64 0.254574 0.0127492
      ];
  case 3
    tit = 'Log-log plot of \nabla^2 (springs) runtime';
    meths = {'tinyad','eigen','tinyremo'};
    D = [
4 8.91924e-06 1.21403e-05 4.56309e-05
8 1.33586e-05 1.88422e-05 7.61199e-05
16 1.31989e-05 2.97308e-05 0.000115161
32 1.96886e-05 8.27599e-05 0.000161161
64 2.3489e-05 0.00016897 0.00020735
128 3.17812e-05 0.000517521 0.000401158
256 6.41155e-05 0.00209329 0.00082716
512 0.000137905 0.00928958 0.00171542
1024 0.000286857 0.0381995 0.00343863
2048 0.000579675 0.155625 0.00701396
4096 0.00123596 0.621884 0.0140005
8192 0.00243354 2.69492 0.0285275
16384 0.00501704 nan 0.0577005
32768 0.010461 nan 0.112178
65536 0.02003 nan 0.222967
     ];
  end
  clf;
  hold on;

  switch pass
  case 1
    Y = 1e-6*D(:,1).^2;
    plot(D(:,1),Y,'--k','LineWidth',1);
    txt([D(end,1) Y(end)],' N^2','FontSize',20);
  case 2
    Y = 2e-7*D(:,1).^3;
    plot(D(:,1),Y,'--k','LineWidth',1,'Color',         0.6*[1.0 0.3 0.2]);
    txt([D(end,1) Y(end)],' N^3','FontSize',20,'Color',0.6*[1.0 0.3 0.2]);

    Y = 7e-10*D(2:end,1).^5;
    plot(D(2:end,1),Y,'--k','LineWidth',1,'Color',     0.6*[0.2 0.3 1.0]);
    txt([D(end,1) Y(end)],' N^5','FontSize',20,'Color',0.6*[0.2 0.3 1.0]);
  case 3
    Y = 2e-6*D(:,1).^1;
    plot(D(:,1),Y,'--k','LineWidth',1,'Color',         0.0*[1.0 0.3 0.2]);
    txt([D(end,1) Y(end)],' N^1','FontSize',20,'Color',0.0*[1.0 0.3 0.2]);

    k = find(isnan(D(:,3)),1);
    Y = 3e-7*D(2:k,1).^2;
    plot(    D(2:k,1),Y,'--k','LineWidth',1,'Color',     0.6*[0.2 0.7 0.3]);
    txt([D(k,1) Y(end)],' N^2','FontSize',20,'Color',0.6*[0.2 0.7 0.3]);
  end

  filt_meths = meths;
  ps = [];
  for mi = numel(meths):-1:1
    switch meths{mi}
    case 'eigen'
      color = [0.2 0.7 0.3];
    case 'tinyremo'
      color = [1.0 0.3 0.2];
    case 'tinyad'
      color = [0.2 0.3 1.0];
      if pass == 1
        filt_meths{mi} = 'tinyad, also computes \nabla^2';
      end
    end
    ps(end+1) = plot(D(:,1),D(:,mi+1),'LineWidth',3,'Color',color);
  end



  hold on;
  set(gca,'YScale','log','XScale','log','FontSize',30,'Color',0.95*[1 1 1]);
  set(gcf,'Color','w');
  legend(ps,flip(filt_meths),'Location','NorthWest','FontSize',30,'Color','w');
  title(tit);
  axis equal;
  xlabel('N');
  ylabel('seconds');

  switch pass
  case 1
    figpng('gradient-llt.png');
  case 2
    figpng('hessian-llt.png');
    case 3
      %axis([1   1.6189e+05   4.5705e-06        5e+4]);
      figpng('hessian-springs.png');
  end
end
