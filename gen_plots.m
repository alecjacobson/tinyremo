for pass = 3
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
  case 3
    tit = 'Log-log plot of \nabla^2 (springs) runtime';
    meths = {'tinyad','eigen','tinyremo'};
    D = [4 4.57048e-06 1.77193e-05 0.00017632
     8 5.91993e-06 5.98598e-05 0.000225291
     16 6.001e-06 0.000133729 0.000306492
     32 8.56161e-06 0.000393698 0.000570621
     64 1.57714e-05 0.00155614 0.00118462
     128 3.05605e-05 0.00617806 0.00246472
     256 6.69575e-05 0.0249318 0.00507625
     512 0.00014017 0.100505 0.0104832
     1024 0.000281135 0.405149 0.0211788
     2048 0.000579278 1.64009 0.0423253
     4096 0.00120139 7.36242 0.084345
     8192 0.00246441 30.0752 0.169259
     16384 0.00528395 nan 0.340507
     32768 0.0102345 nan 0.680715
     65536 0.019537 nan 1.3697];
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
