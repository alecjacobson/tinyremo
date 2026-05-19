for pass = 1:2
  switch pass
  case 1
    tit = 'Log-log plot of \nabla (llt) runtime';
    meths = {'eigen','tinyad','tinyremo'};
    D = [
      2 1.27077e-06 1.28746e-07 2.2316e-06
      4 3.9506e-06 2.52962e-06 6.22988e-06
      8 2.31385e-05 3.81207e-05 2.83694e-05
      16 0.000103152 0.000603151 0.00012917
      32 0.000834198 0.0145271 0.000771151
      64 0.00819852 0.458206 0.00383134
      ];
  case 2
    tit = 'Log-log plot of \nabla^2 (llt) runtime';
    meths = {'tinyad','tinyremo'};
    D = [
      2 1.3113e-07 3.86e-06
      4 1.85966e-06 1.61004e-05
      8 3.34692e-05 0.00016197
      16 0.000461631 0.000792902
      32 0.0120034 0.00541788
      64 0.428727 0.0336399
      ];
  case 3
    tit = 'Log-log plot of \nabla^2 (springs) runtime';
    meths = {'tinyad','eigen','tinyremo'};
    D = [
     4 1.69992e-06 3.60966e-06 1.88589e-05
     8 3.00169e-06 9.53913e-06 3.932e-05
     16 5.09024e-06 2.47908e-05 7.81584e-05
     32 1.01686e-05 8.17204e-05 0.000159111
     64 1.84202e-05 0.000265021 0.00032928
     128 3.72171e-05 0.00100518 0.00141552
     256 0.000191641 0.00789808 0.00208768
     512 0.000281413 0.0241442 0.00385525
     1024 0.000350316 0.0882595 0.00748217
     2048 0.000848929 0.368213 0.0135747
     4096 0.00157344 1.03556 0.0256414
     8192 0.00297356 nan 0.049479
     16384 0.00566757 nan 0.0963985
     32768 0.0124495 nan 0.20086
     65536 0.0217559 nan 0.390071
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
