
    elif 1 == 2:
#
#       do with start. implement SA guy start
#
        res=dual_annealing(simu_obp,args=(n,pnts,maps,c_l,c_r,c_v_0,0),bounds=opt_0i_bds,\
            callback=partial(back_da,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)),\
            seed=1,no_local_search=False)
    elif 2 == 1:
        res=differential_evolution(simu_obp,args=(n,pnts,maps,c_l,c_r,c_v_0,0),bounds=opt_0_bds,\
            callback=partial(back_de,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)),\
            seed=1,workers=cpus,updating='deferred',polish=False,disp=False,integrality=opt_0_its)
    elif 1 == 2:
        res=differential_evolution(simu_obp,args=(n,pnts,maps,c_l,c_r,c_v_0,0),bounds=opt_0_bds,\
            callback=partial(back_de,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)),\
            seed=1,workers=cpus,updating='deferred',polish=False,disp=False)
    elif 1 == 2:
        res=shgo(simu_obp,args=(n,pnts,maps,c_l,c_r,c_v_0,0),bounds=opt_0_bds,\
            callback=partial(back_x,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)),\
            workers=cpus,options={'disp': True})
    elif 1 == 2:
        res=direct(simu_obp,args=(n,pnts,maps,c_l,c_r,c_v_0,0),bounds=opt_0_bds,\
            callback=partial(back_x,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)))
    elif 1 == 1:
        res=brute(simu_obp,args=(n,pnts,maps,c_l,c_r,c_v_0,0),ranges=opt_0_bds,workers=cpus)
#           callback=partial(back_x,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)))
    else:
#
#       variables can not be bounded.... use modulo function... it has a derivative
#
        res=dual_annealing(simu_obp,args=(n,pnts,maps,c_l,c_r,c_v_0,0),bounds=opt_0_bds,\
            callback=partial(back_da,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)),
            seed=1,maxiter=1,no_local_search=False)
        xi=res.x
        res=basinhopping(simu_obp,minimizer_kwargs={"method":"Nelder-Mead","args":(n,pnts,maps,c_l,c_r,c_v_0,0)},\
            callback=partial(back_bh,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)),\
            seed=1,x0=xi)
