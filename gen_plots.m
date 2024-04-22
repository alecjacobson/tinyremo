for pass = 1:2
  switch pass
  case 1
    tit = 'Log-log plot of \nabla (llt) runtime';
    meths = {'eigen','tinyad','tinyremo'};
    D = [2 6.64949e-06 2.69413e-07 1.8096e-06
    4 1.92308e-05 1.76907e-06 5.24044e-06
    8 7.81894e-05 2.8162e-05 2.04587e-05
    16 0.000185301 0.000427279 9.161e-05
    32 0.000949619 0.00984386 0.00058212
    64 0.00592546 0.337963 0.00258134];
  case 2
    tit = 'Log-log plot of \nabla^2 (llt) runtime';
    meths = {'tinyad','tinyremo'};
    D = [2 1.09673e-07 5.21898e-06
    4 1.76907e-06 2.44403e-05
    8 2.75302e-05 0.000179689
    16 0.000432031 0.00114989
    32 0.00994142 0.0121569
    64 0.347589 0.109768];
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
  end
end
