for pass = 1:3
  switch pass
  case 1
    tit = 'Log-log plot of \nabla (llt) runtime';
    meths = {'eigen','tinyad','tinyremo'};
    D = [
2 1.61886e-06 2.5034e-07 9.91821e-07
4 5.76973e-06 1.15871e-06 3.05891e-06
8 3.31306e-05 1.95718e-05 1.66392e-05
16 0.00014071 0.00025831 5.66411e-05
32 0.000597701 0.00647487 0.000307629
64 0.00381919 0.246406 0.00186759
128 0.0113789 nan 0.00256498
      ];
  case 2
    tit = 'Log-log plot of \nabla^2 (llt) runtime';
    meths = {'tinyad','tinyremo'};
    D = [
2 8.10623e-08 1.61171e-06
4 1.17064e-06 6.82831e-06
8 1.94693e-05 5.46718e-05
16 0.00025733 0.000311558
32 0.00664648 0.00239789
64 0.251761 0.0153635
      ];
  case 3
    tit = 'Log-log plot of \nabla^2 (springs) runtime';
    meths = {'tinyad','eigen','tinyremo'};
    D = [
4 5.18084e-06 5.67913e-06 2.93612e-05
8 7.25985e-06 1.22404e-05 5.39804e-05
16 1.12009e-05 2.70605e-05 9.81688e-05
32 1.67894e-05 6.59418e-05 0.000130959
64 2.29001e-05 0.000148151 0.000189979
128 3.28588e-05 0.000475082 0.000349517
256 6.51646e-05 0.0019514 0.000719757
512 0.000141005 0.00865475 0.00146091
1024 0.00028948 0.0367785 0.0029515
2048 0.000583967 0.149089 0.00606537
4096 0.00125158 0.595479 0.0131345
8192 0.002424 2.49224 0.0251236
16384 0.00506854 nan 0.049721
32768 0.0101739 nan 0.0980386
65536 0.0197406 nan 0.196821
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
