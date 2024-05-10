import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.optimize import newton
from scipy.stats import norm
from scipy.optimize import least_squares


class smile_K_interp:

    def SABR_calibration(K,t,beta,iv,iv_ATM,K_ATM,f):
        '''
        Calibration of SABR parameters rho and nu (here denominated gamma)

        Input: 
        - K: set of strikes 
        - t: tenors 
        - beta: SABR beta
        - iv: implied volatility 
        - iv_ATM: implied volatility ATM 
        - K_ATM: strike ATM 
        - f: forward value 

        Output: 
        - Calibrated_Parms: return calibrated parameters rho and nu in dictionary 
        '''
        # ensure that K and iv are array 
        K = np.array(K)
        iv = np.array(iv)

        # define objective function needed for minimation problem  (minimize for rho and gamma)
        f_obj = lambda x: smile_K_interp.f_obj(beta,x[0],x[1],iv,K,iv_ATM,K_ATM,f,t)
        
        # initial guess
        initial = np.array([-0.8,0.4])
        # minimization problem 
        pars  = minimize(f_obj,initial,method='nelder-mead', options = {'xatol': 1e-15, 'disp': False})
        
        # store parameter estimations in dictionary afterwards returned as function output 
        rho_est = pars.x[0]
        gamma_est  = pars.x[1]
        Calibrated_Parms =  {"rho":rho_est,"gamma":gamma_est}
        return Calibrated_Parms

    def f_obj(beta,rho,gamma,iv,K,iv_ATM,K_ATM,f,t):
        '''
        Return the error that needs to be minimized for the correct calibration of the SABR model 

        Input: 
        - beta,rho,gamma: SABR parameters 
        - iv: implied vol array 
        - K: strikes array 
        - iv_ATM: implied vol ATM options 
        - K_ATM: strike ATM option 
        - f: forward FX rate 
        - t: tenor

        Output: 
        - value: norm of the error vector (difference between SABR smile and market smile)
        '''

        # check that iv is np array 
        if iv is not np.array:
            iv = np.array(iv).reshape([len(iv),1])
        
        # retrive optimal alpha 
        alpha_est = smile_K_interp.opt_alpha(iv_ATM, K_ATM,t,f,beta,rho,gamma)
    
        # Error is defined as a difference between the market and the model
        errorVector = smile_K_interp.SABR_vol(K,t,f,alpha_est,beta,rho,gamma) - iv
        
        # Target value is a norm of the ErrorVector
        value = np.linalg.norm(errorVector)  

        return value

    def opt_alpha(iv_ATM, K_ATM,t,f,beta,rho,gamma):
        '''
        Return optimal value of alpha (SABR parameter)

        Input: 
        - iv_ATM: implied vol ATM options 
        - K_ATM: strike ATM option 
        - f: forward FX rate 
        - t: tenor
        - beta,rho,gamma: SABR parameters 

        Output: 
        - alpha_est: alpha SABR parameter estimation  
        '''

        # define objective function that will be used in Newton interative process
        # Here we want to optimize the SABR vol function for alpha 
        target = lambda alpha: smile_K_interp.SABR_vol(K_ATM,t,f,alpha,beta,rho,gamma)-iv_ATM

        # Newton given initial guess for alpha and tolerance
        alpha_est = newton(target,0.5,tol=0.0000001)
        return alpha_est

    def SABR_vol(K,T,f,alpha,beta,rho,gamma):
        '''
        Return SABR volatility 

        Input: 
        - alpha,beta,rho,gamma: SABR parameters 
        - K: strike array 
        - T: tenor array 
        - f: forward FX array

        Output:
        - impVol: SABR volatility 
        '''
        #We make sure that the input is of array type
        if gamma == 0 :
            return 0.0
        if type(K) == float:
            K = np.array([K])
        if K is not np.array:
            K = np.array(K).reshape([len(K),1])

        # compute all the block inside SABR vol formula     
        z        = gamma/alpha*np.power(f*K,(1.0-beta)/2.0)*np.log(f/K)
        x_z      = np.log((np.sqrt(1.0-2.0*rho*z+z*z)+z-rho)/(1.0-rho))
        A        = alpha/(np.power(f*K,((1.0-beta)/2.0))*(1.0+np.power(1.0-\
                        beta,2.0)/24.0*
                                np.power(np.log(f/K),2.0)+np.power((1.0-beta),\
                                            4.0)/1920.0*
                                np.power(np.log(f/K),4.0)))
        B1       = 1.0 + (np.power((1.0-beta),2.0)/24.0*alpha*alpha/(np.power((f*K),
                    1-beta))+1/4*(rho*beta*gamma*alpha)/(np.power((f*K),
                                ((1.0-beta)/2.0)))+(2.0-3.0*rho*rho)/24.0*gamma*gamma)*T
        impVol   = A*(z/x_z) * B1
        return impVol
    
    def smile_K_interp_SABR(smile_df,tenor_str,expiry,S,rd,rf,delta, delta_target, w, plot=False):
        '''
        Interpolation of the smile across strikes for each maturity

        Input:
        - smile_df: market smile dataframe
        - tenor_str: all tenors in string format 
        - expiry: all tenors in float format 
        - S: spot fx 
        - rd, rf: risk free rates (domestic and foreign)
        - delta: all market deltas 
        - delta_target:
        - w: 1 for call and -1 for put
        - plot: plot the interpolated smile for each maturity (default = False)

        Output:
        - smile interp: vol surface dataframe of interpolated implied vol for each tenor/interpolated strike pair 
        - smile_mrk: dataframe with SABR calibration data for each market tenor 
        - mrk_dfs: list of dfs ordered by their tenor with same info of smile_mrk 
        '''

        # define maximum range of strikes among all the tenors (max in last smile with highest tenor)
        k = smile_df[smile_df['Tenor_str'] == tenor_str[-1]].iloc[:,7:10]
        put_k = k['K_Put'].iloc[::-1].tolist()
        atm_k = [k.iloc[0, 0]]
        call_k = k['K_Call'].tolist()
        # concatenate market strikes 
        K_i = np.array(put_k + atm_k + call_k)

        # define interpolated grid of strikes 
        K_grid = np.array([np.linspace(K_i[i],K_i[i+1],200, endpoint=False) for i in range(len(K_i)-1)]).flatten().tolist()
        K_grid.append(K_i[-1])

        # retrive all the market strikes found while building the smile and merge with the interpolated ones 
        for i in tenor_str:
            k = smile_df[smile_df['Tenor_str'] == i].iloc[:,7:13]
            put_k = k['K_Put'].iloc[::-1].tolist()
            atm_k = [k.iloc[0, 0]]
            call_k = k['K_Call'].tolist()
            K_i = put_k + atm_k + call_k
            K_grid.extend(K_i)

        # sort strikes 
        K_grid_i = sorted(set(K_grid))

        # declare empty list used in the following for loop 
        mrk_dfs = []
        interpolated_iv_dfs1 = []
        gamma_pars = []
        rho_pars = []
        alpha_pars = []

        # for each tenor interpolate the volatility values across strikes defined in K_grid_i
        for tenor_i in range(len(tenor_str)):

            # select rows with same tenor 
            k_iv = smile_df[smile_df['Tenor_str'] == tenor_str[tenor_i]].iloc[:,7:13]

            # define set of initial strikes and corresponding volatility obtained from the smile construction 
            put_k_col = k_iv['K_Put'].iloc[::-1].tolist()
            atm_k_col = [k_iv.iloc[0, 0]]
            call_k_col = k_iv['K_Call'].tolist()
            # concatenate strikes 
            K = np.array(put_k_col + atm_k_col + call_k_col)

            # concatenate related market smile volatilities 
            put_v_col = k_iv['Vol_Put'].iloc[::-1].tolist()
            atm_v_col = [k_iv.iloc[0, 3]]
            call_v_col = k_iv['Vol_Call'].tolist()
            IV = np.array(put_v_col + atm_v_col + call_v_col)

            # save initial market data (market strikes and volatility pairs)
            data_mkt = {'Tenor': expiry[tenor_i], 'K': K, 'IV': IV}
            df_mkt = pd.DataFrame(data_mkt)
            # append each market df in list 
            mrk_dfs.append(df_mkt)

            # store maturity value 
            Tt = expiry[tenor_i]

            # set beta = 1 
            beta = 1

            # Select ATM values 
            Atm_IV = atm_v_col
            Atm_K = atm_k_col

            # initialize forward 
            Fwd = S[tenor_i] * np.exp((rd[tenor_i] - rf[tenor_i]) * Tt)

            # Hegan calibration: retrive calibrated gamma, rho, alpha 
            Calibrated_Parms = smile_K_interp.SABR_calibration(K,Tt,beta,IV,Atm_IV,Atm_K,Fwd)
            gamma = Calibrated_Parms.get('gamma') # get gamma par
            rho = Calibrated_Parms.get('rho') # get rho par 
            alpha = smile_K_interp.opt_alpha(Atm_IV,Atm_K,Tt,Fwd,beta,rho,gamma)[0][0]

            # store parameters in lists 
            gamma_pars.append(gamma)
            rho_pars.append(rho)
            alpha_pars.append(alpha)
            
            # SABR volatility on interpolated strikes 
            iv_Hagan1 = smile_K_interp.SABR_vol(K_grid_i,Tt,Fwd,alpha,beta,rho,gamma)
            iv_Hagan1 = iv_Hagan1.flatten().tolist()

            # store results on interpolated strikes and interpolated volatilities in dataframe 
            data = {'Strike': K_grid_i, str(expiry[tenor_i]): iv_Hagan1}
            df = pd.DataFrame(data)
            df.set_index('Strike', inplace=True)
            interpolated_iv_dfs1.append(df)

            # visual output during the processs 
            print('Optimal par for tenor {0}: alpha = {1}  -  gamma = {2}  -  rho = {3}'.format(tenor_str[tenor_i],alpha,gamma,rho))
            # plot the smile for each market tenor if plot set True
            if plot == True:
                # plot result
                plt.figure(figsize=(8, 6))
                plasma_cmap = plt.cm.plasma
                plt.plot(df_mkt.K,df_mkt.IV,'o', color='black', markersize=3 , label='Market values')
                num_tenors = len(tenor_str)
                color_index = 0.85- tenor_i / num_tenors  # Normalized index (between 0 and 1)
                color = plasma_cmap(color_index)
                plt.plot(K_grid_i, iv_Hagan1, color=color, label='SABR interp')#label=tenor_str[tenor_i]
                plt.plot(Atm_K, Atm_IV, '*', color='orange', label='ATM level')
                plt.grid()
                ax_font = {'fontname': 'Times New Roman', 'fontsize': 11}
                plt.xlabel('Strike', fontdict=ax_font)
                plt.ylabel('Volatility', fontdict=ax_font)
                legend_font = {'family': 'Times New Roman', 'size': 10}  # Customize font family and size as desired
                plt.legend(prop=legend_font)
                title_font = {'fontname': 'Times New Roman', 'fontsize': 12}
                title = 'SABR smile ({0})'.format(tenor_str[tenor_i])
                plt.title(title, fontdict=title_font)
                description = 'SABR pars: alpha = {0} ~ gamma = {1} ~ rho = {2} ~ beta = {3}'.format(alpha, gamma, rho, 1.0)
                plt.figtext(0.5, 0.01, description, ha='center', fontdict=ax_font)
                plt.show();

        # store all the interpolated values in a dataframe 
        # containing strikes and implied volatilities pairs for each maturity 
        smile_interp = interpolated_iv_dfs1[0]

        for df in interpolated_iv_dfs1[1:]:
            smile_interp = pd.concat([smile_interp, df], axis=1)

        # enrich market iv-K dataframe with all the SABR parameters (usefull later for tenor interpolation)
        spot = S[0]
        rd_set = rd[0:8]
        rf_set = rf[0:8]

        for mrk_df_index in range(len(mrk_dfs)):
            mrk_dfs[mrk_df_index]['w'] = w
            mrk_dfs[mrk_df_index]['delta_target'] = delta_target
            mrk_dfs[mrk_df_index]['S'] = spot
            mrk_dfs[mrk_df_index]['rd'] = rd_set[mrk_df_index]
            mrk_dfs[mrk_df_index]['rf'] = rf_set[mrk_df_index]
            mrk_dfs[mrk_df_index]['Fwd'] = spot*np.exp((rd_set[mrk_df_index]- rf_set[mrk_df_index]) * mrk_dfs[mrk_df_index]['Tenor'])
            mrk_dfs[mrk_df_index]['beta'] = 1
            mrk_dfs[mrk_df_index]['gamma'] = gamma_pars[mrk_df_index]
            mrk_dfs[mrk_df_index]['alpha'] = alpha_pars[mrk_df_index]
            mrk_dfs[mrk_df_index]['rho'] = rho_pars[mrk_df_index]

        smile_mrk = mrk_dfs[0]

        # concatenate df in list (mrk_dfs) inside a unique dataframe
        for df in mrk_dfs[1:]:
            smile_mrk = pd.concat([smile_mrk, df], axis=0)

        return smile_interp, smile_mrk, mrk_dfs
    
    def plot_surf_K_interp(smile_interp_df):
        # plot 3D surface 
        X = np.array(smile_interp_df.index)
        Y = smile_interp_df.columns.to_numpy().astype(float) 
        X, Y = np.meshgrid(X, Y)
        Z = smile_interp_df.values.T 
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='plasma')
        ax.set_xlabel('Strike')
        ax.set_ylabel('Tenor')
        ax.set_title('VOLATILITY SURFACE')
        ax.view_init(elev=30, azim=45)
        plt.show()
    
    def plot3D_surf_K_interp(smile_interp_df):
        # plot dynamic 3D surface 
        X = np.array(smile_interp_df.index)
        Y = smile_interp_df.columns.to_numpy().astype(float) 
        Z = smile_interp_df.values.T  # Transpose Z to match the shape expected by plot_surface
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
        fig.update_layout(
            title='3D Plot',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'))
        fig.show()

class smile_T_interp:

    def SABR_vol(K,T,f,alpha,beta,rho,gamma):

        '''
        Return SABR volatility 

        Input: 
        - alpha,beta,rho,gamma: SABR parameters 
        - K: strike array 
        - T: tenor array 
        - f: forward FX array

        Output:
        - impVol: SABR volatility 
        '''

        if gamma == 0 :
            return 0.0
        if type(K) == float:
            K = np.array([K])
        if K is not np.array:
            K = np.array(K).reshape([len(K),1])
            
        z        = gamma/alpha*np.power(f*K,(1.0-beta)/2.0)*np.log(f/K)
        x_z      = np.log((np.sqrt(1.0-2.0*rho*z+z*z)+z-rho)/(1.0-rho))
        A        = alpha/(np.power(f*K,((1.0-beta)/2.0))*(1.0+np.power(1.0-\
                        beta,2.0)/24.0*
                                np.power(np.log(f/K),2.0)+np.power((1.0-beta),\
                                            4.0)/1920.0*
                                np.power(np.log(f/K),4.0)))
        B1       = 1.0 + (np.power((1.0-beta),2.0)/24.0*alpha*alpha/(np.power((f*K),
                    1-beta))+1/4*(rho*beta*gamma*alpha)/(np.power((f*K),
                                ((1.0-beta)/2.0)))+(2.0-3.0*rho*rho)/24.0*gamma*gamma)*T
        impVol   = A*(z/x_z) * B1
        return impVol
    
    def fwd_delta(sigma,K,T,w,S,rd,rf): 
        '''
        Find the forward delta of an option

        Input: 
        - sigma: volatility
        - K: stike (guessed in 'MS_K')
        - T: tenor
        - w: 1.0 for call and -1.0 for put 
        - S: spot fx 
        - rd, rf: risk free rates (domestic and foreign)
        - delta_convention: types of delta convention used 

        Output: 
        - option forward delta
        '''
        d1 = (np.log(S / K) + (rd - rf + .5 * sigma * sigma) * T) / (sigma*np.sqrt(T))
        return w * norm.cdf(w * d1) # page 67 clark

    def flat_fwd_interp(T, t1, t2, sigma1, sigma2): 
        
            '''
            Perform backbone flat forward interpolation across maturities 

            Input: 
            - T: intermediate tenor
            - t1: lower market tenor 
            - t2: higher market tenor
            - sigma1: implied volatility at lower market tenor 
            - sigma2: implied volatility at higher market tenor 

            Output: 
            - sigmaT: flat forward interpolated volatility
            '''
            # backbone flat forward interpolation formula
            sigmaT = np.sqrt(( t2*(T-t1)/(T*(t2-t1)) ) * sigma2**2 + ( t1*(t2-T)/(T*(t2-t1)) ) * sigma1**2)
            return sigmaT

    def delta_diff(K, T, f, alpha, beta, rho, gamma, w, S, rd, rf, target_delta):

        '''
        Find the absolute difference between market delta and SABR delta  

        Input: 
        - K: array of initial market strikes   
        - T: array of market tenors  
        - f: array of forward values 
        - alpha,beta,rho,gamma: SABR parameters 
        - w: 1.0 for call and -1.0 for put
        - S: spot fx 
        - rd, rf: risk free rates (domestic and foreign) 
        - target_delta: array of all market option deltas

        Output:
        - abs_diff: absolute difference between market delta and calculated delta 
        '''
        # find the implied volatility 
        sigma = smile_T_interp.SABR_vol(K, T, f, alpha, beta, rho, gamma)
        # calculate forward delta
        calculated_delta = smile_T_interp.fwd_delta(sigma, K, T, w, S, rd, rf)
        # define the absolute difference between market delta and previously calculated delta 
        abs_diff = abs(calculated_delta - target_delta)
        
        return abs_diff
    
    def adj_K_IV(mrk_dfs, tenor):

        '''
        Find the adjusted strikes and volatilities given the artificial conventions imposed  

        Input: 
        - mrk_dfs: list of dataframe containing smile data for each tenor  
        - tenor: array of market tenors  

        Output:
        - k_hat_dfs: dataframe with adjusted strikes and volatilities  
        '''
        # define boundaries for strikes 
        K_bounds = (0.6, 2.7) 

        # for each smile df contained in the list mrk_dfs
        for mrk in mrk_dfs:
            K_optimal_list = []
            for r in mrk.index:
                K_initial_guess = mrk.iloc[r, 1]

                # Use minimize to find the optimal K
                result = minimize(smile_T_interp.delta_diff, K_initial_guess, method='nelder-mead', 
                                  options = {'xatol': 1e-15, 'disp': False}, 
                                  args=(mrk.loc[r,'Tenor'], mrk.loc[r,'Fwd'], mrk.loc[r,'alpha'], 
                                        mrk.loc[r,'beta'], mrk.loc[r,'rho'], mrk.loc[r,'gamma'], 
                                        mrk.loc[r,'w'], mrk.loc[r,'S'], mrk.loc[r,'rd'], 
                                        mrk.loc[r,'rf'], mrk.loc[r,'delta_target']), bounds=[K_bounds])

                # Extract the optimal strike K from the result
                K_optimal = result.x[0]
                K_optimal_list.append(K_optimal)
            # strore adjusted strikes 
            mrk['K_^'] = K_optimal_list
            # calculate adjusted volatility 
            new_hegan_iv = smile_T_interp.SABR_vol(mrk['K_^'],mrk.loc[0,'Tenor'],mrk.loc[0,'Fwd'],
                                                   mrk.loc[0,'alpha'],mrk.loc[0,'beta'],mrk.loc[0,'rho'],
                                                   mrk.loc[0,'gamma'])
            # strore adjusted vol 
            mrk['IV_^'] = new_hegan_iv
        
        # manage data 
        k_hat_dfs = []
        for i in range(len(mrk_dfs)):

            k_hat_df = pd.DataFrame(index=mrk_dfs[i].index)
            tenor = mrk_dfs[i].loc[0, 'Tenor']
            # k_hat_df['Tenor'] = mrk_dfs[i]
            if tenor < 4.0:
                k_hat_df = pd.concat([mrk_dfs[i]['Tenor'], mrk_dfs[i].iloc[:, [-2,-1]], mrk_dfs[i].iloc[:, 3:9]], axis=1)
            else:
                k_hat_df = mrk_dfs[i].iloc[:, 0:9]
            k_hat_df.columns = ['Tenor', 'K_^', 'IV_^', 'w', 'delta_target', 'S', 'rd', 'rf', 'Fwd']
            k_hat_dfs.append(k_hat_df)

        return k_hat_dfs
    
    def surf_T_interp(initial_tenors, interp_tenors, mrk_df_list):
        
        '''
        Returns the flat forward interpolated data

        Input: 
        - initial_tenors: array of initial market tenors   
        - interp_tenors: array of intermediate tenors for interpolation 
        - mrk_df_list: list of all market smile dataframes  

        Output:
        - interp_df_list: list of interpolated tenor data  
        - K_interp_entire_list: list of interpolated and market tenor data 
        '''

        K_interp_entire_list = []
        interp_df_list = []
        interp_t_index = 0

        # for each market tenor
        for t_i in range(len(initial_tenors)):
            # set t2 = tenor at time t+1 
            t2 = initial_tenors[t_i]
            # set t2 = tenor at time t-1 
            t1 = initial_tenors[t_i-1] 
            
            # utill the intermediate tenor remains inside two market tenors 
            while interp_tenors[interp_t_index] < initial_tenors[t_i]:
                
                # select all the data needed 
                T = interp_tenors[interp_t_index]

                tenor_interp_iv = []
                tenor_interp_k = []
                tenor_interp_T = []

                for i in range(len(mrk_df_list[t_i].index)): 
                    sigma1 = mrk_df_list[t_i-1].iloc[i,2]
                    sigma2 = mrk_df_list[t_i].iloc[i,2] 

                    w = mrk_df_list[t_i].iloc[i,3]  
                    target_delta = mrk_df_list[t_i].iloc[i,4]  
                    S = mrk_df_list[t_i].iloc[i,5]
                    rd = mrk_df_list[t_i].iloc[i,6]  
                    rf = mrk_df_list[t_i].iloc[i,7]
                    
                    k1= mrk_df_list[t_i-1].iloc[i,1]
                    k2= mrk_df_list[t_i].iloc[i,1]
                    k_bounds = (min(k1,k2), max(k1,k2))

                    # Perform least squares optimization on K given this objective function 
                    def objective(K):
                        
                        # perform flat forward interpolation 
                        sigmaT = smile_T_interp.flat_fwd_interp(T, t1, t2, sigma1, sigma2)
                        # return the delta given volatility obtained in interpolation 
                        return smile_T_interp.fwd_delta(sigmaT, K, T, w, S, rd, rf) - target_delta

                    # Residual function for least squares optimization
                    def residual(K):
                        return abs(objective(K))#**2  # Squared residuals for least squares
                    
                    # strore optimal K found from the optimization problem 
                    optimal_k = least_squares(residual, x0=[min(k1,k2)], bounds=k_bounds).x[0]
                    tenor_interp_k.append(optimal_k)

                    # store interpolated sigma 
                    interpolated_sigma = smile_T_interp.flat_fwd_interp(T, t1, t2, sigma1, sigma2)
                    tenor_interp_iv.append(interpolated_sigma)

                    tenor_interp_T.append(T)
                
                # store all the info in a dataframe 
                interp_df = pd.DataFrame()
                interp_df['Tenor'] = tenor_interp_T
                interp_df['K_^'] = tenor_interp_k  
                interp_df['IV_^'] = tenor_interp_iv
                interp_df = pd.concat([interp_df, mrk_df_list[t_i-1].iloc[:,-6:]], axis=1)
                interp_df_list.append(interp_df)
                interp_t_index = interp_t_index +1
                K_interp_entire_list.append(tenor_interp_k)
                
                if interp_t_index==len(interp_tenors):
                    break 
            
            interp_df_list.append(mrk_df_list[t_i])
            K_interp_entire_list.append(mrk_df_list[t_i].iloc[:,1].tolist())
            
            # visual output for precedure 
            if t1 != 3.0:
                print('Interpolation in range [',t1, ',', t2 , '] completed')

        K_interp_entire_list = np.unique(np.sort(np.array(K_interp_entire_list).flatten()))

        return interp_df_list, K_interp_entire_list
    
    def SABR_K_on_interpT(K_grid,interp_df_list, all_tenors, initial_tenors, initil_tenor_str, plot=False):
        
        '''
        Returns SABR volatility on the interpolated strikes give the interpolated smiles 

        Input: 
        - K_grid: array of strikes on which interpolation is performed    
        - interp_df_list: list of smile df containing info obtained from tenor interpolation process 
        - all_tenors: all tenors  (market + interpolated)  
        - initial_tenors: initial tenors only (market)
        - initil_tenor_str: list of initial market tenors in string format 
        - plot: set to True if you want the plot of the market smiles under the new artificial convention

        Output:
        - interp_iv: dataframe containig the entirly populated volatility matrix (stike and tenor interpolation)
        - interp_iv_par: same info as interp_iv + SABR parameters for each intermediate and market tenor
        '''

        intepolated_iv_dfs = []
        interp_par_dfs = []
        gamma_par = []
        rho_par = []
        alpha_par = []

        for tenor_i in range(len(all_tenors)):

            # initalize set of initial strikes and corresponding volatility 
            Atm_K = [interp_df_list[tenor_i].loc[5,'K_^']]
            K = np.array(interp_df_list[tenor_i].loc[:,'K_^'].tolist())
            Atm_IV = [interp_df_list[tenor_i].loc[5,'IV_^']]
            IV = np.array(interp_df_list[tenor_i].loc[:,'IV_^'].tolist())

            # initialize maturity (float)
            Tt = all_tenors[tenor_i]
            if Tt != interp_df_list[tenor_i].loc[0,'Tenor']:
                print('Error')

            # set beta = 1 
            beta = 1

            # initialize forward 
            Fwd = np.array(interp_df_list[tenor_i].loc[0,'Fwd'].tolist())

            # Hegan calibration 
            Calibrated_Parms = smile_K_interp.SABR_calibration(K,Tt,beta,IV,Atm_IV,Atm_K,Fwd)
            gamma = Calibrated_Parms.get('gamma')
            rho = Calibrated_Parms.get('rho')
            alpha = smile_K_interp.opt_alpha(Atm_IV,Atm_K,Tt,Fwd,beta,rho,gamma)[0][0]

            gamma_par.append(gamma)
            rho_par.append(rho)
            alpha_par.append(alpha)
            
            # calibrated hegan on interpolated strikes 
            iv_Hagan1 = smile_K_interp.SABR_vol(K_grid,Tt,Fwd,alpha,beta,rho,gamma)
            iv_Hagan1 = iv_Hagan1.flatten().tolist()

            data = {'Strike': K_grid, str(all_tenors[tenor_i]): iv_Hagan1}
            df = pd.DataFrame(data)
            df.set_index('Strike', inplace=True)
            intepolated_iv_dfs.append(df)
            
            data_par = {'Tenor':all_tenors[tenor_i], 'K': K_grid, 'IV': iv_Hagan1}
            df_par = pd.DataFrame(data_par)
            w = [-1 if K_grid[i]< Atm_K else 1 for i in range(len(K_grid))]
            df_par['w'] = w 
            
            df_par['S'] = np.full_like(K_grid, interp_df_list[tenor_i].iloc[0,5])
            df_par['rd'] = np.full_like(K_grid, interp_df_list[tenor_i].iloc[0,6])
            df_par['rf'] = np.full_like(K_grid, interp_df_list[tenor_i].iloc[0,7])
            df_par['gamma'] = gamma
            df_par['alpha'] = alpha
            df_par['rho'] = rho
            df_par['beta'] = 1

            interp_par_dfs.append(df_par)
            # visual inspection of the process (printed)
            print('Opt pars for tenor {0}: alpha={1}, gamma = {2} and rho = {3}'.format(all_tenors[tenor_i],alpha,gamma,rho))
            
            # plot 
            if all_tenors[tenor_i] in initial_tenors and plot == True:

                plt.figure(figsize=(8, 6))
                plasma_cmap = plt.cm.plasma
                plt.plot(K,IV,'o', color='black', markersize=3 , label='Market values')
                num_tenors = len(initil_tenor_str)
                color_index = 0.85- tenor_i / num_tenors  # Normalized index (between 0 and 1)
                color = plasma_cmap(color_index)
                plt.plot(K_grid, iv_Hagan1, color=color, label='SABR interp')#label=tenor_str[tenor_i]
                plt.plot(Atm_K, Atm_IV, '*', color='orange', label='ATM level')
                plt.grid()
                ax_font = {'fontname': 'Times New Roman', 'fontsize': 11}
                plt.xlabel('Strike', fontdict=ax_font)
                plt.ylabel('Volatility', fontdict=ax_font)
                legend_font = {'family': 'Times New Roman', 'size': 10}  # Customize font family and size as desired
                plt.legend(prop=legend_font)
                title_font = {'fontname': 'Times New Roman', 'fontsize': 12}
                title = 'SABR smile ({0})'.format(all_tenors[tenor_i])
                plt.title(title, fontdict=title_font)
                description = 'SABR pars: alpha = {0} ~ gamma = {1} ~ rho = {2} ~ beta = {3}'.format(alpha, gamma, rho, 1.0)
                plt.figtext(0.5, 0.01, description, ha='center', fontdict=ax_font)
                plt.show();
        
        # store data of fully populated voaltility matrix (tenor and strike interpolation)
        interp_iv = intepolated_iv_dfs[0]
        for df in intepolated_iv_dfs[1:]:
            interp_iv = pd.concat([interp_iv, df], axis=1)

        # store interpolated values and SABR parameters
        interp_iv_par = interp_par_dfs[0]
        for par in interp_par_dfs[1:]:
            interp_iv_par = pd.concat([interp_iv_par, par], axis=0)

        return interp_iv, interp_iv_par
    
    def IV_Surf_Smoothing(mrk_dfs,initial_tenors,interp_tenors,initil_tenor_str, plot=False):
        
        '''
        Returns interpolated smiles across strikes and tenors  

        Input: 
        - mrk_dfs: market volatility surface dataframe 
        - initial_tenors: list of initial tenors 
        - interp_tenors: listo of all the interpolated tenors 
        - initial_tenors: initial tenors only (market)
        - initil_tenor_str: list of initial market tenors in string format 
        - plot: set to True if you want the plot of the market smiles under the new artificial convention

        Output:
        - k_hat_dfs: market smiles calibrated with SABR under new artificial market conventions
        - interp_df_list: list of all the interpolated df in tenor space 
        - interp_iv: final interpolated vol surface (strike/tenors)
        - interp_iv_par: dataframe containing info on interpolated vol surface and on SABR parameters
        '''

        # find the adjusted strikes and volatilitites with the new artificial convention 
        k_hat_dfs = smile_T_interp.adj_K_IV(mrk_dfs, initial_tenors)
        # interpolate in tenors 
        interp_df_list,  K_interp_entire_list = smile_T_interp.surf_T_interp(initial_tenors, interp_tenors, k_hat_dfs)

        # interpolate in strikes 
        all_tenors = np.unique(np.concatenate((interp_tenors, initial_tenors)))
        k_start = interp_df_list[-1].iloc[0, 1]
        k_end = interp_df_list[-1].iloc[-1, 1]
        K_grid1 = np.linspace(k_start,k_end,1000, endpoint=True)
        K_grid = np.unique(np.concatenate((K_interp_entire_list, K_grid1)))
        interp_iv, interp_iv_par = smile_T_interp.SABR_K_on_interpT(K_grid,interp_df_list, all_tenors, initial_tenors, initil_tenor_str, plot=plot)

        return k_hat_dfs, interp_df_list, interp_iv, interp_iv_par
    
    def plot_surf_KT_interp(smile_interp_df):

        # plot 3D surface 
        X = np.array(smile_interp_df.index)
        Y = smile_interp_df.columns.to_numpy().astype(float) 
        X, Y = np.meshgrid(X, Y)
        Z = smile_interp_df.values.T 
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='plasma')
        ax.set_xlabel('Strike')
        ax.set_ylabel('Tenor')
        ax.set_title('VOLATILITY SURFACE')
        ax.view_init(elev=30, azim=45)
        plt.show()

    def plot3D_surf_KT_interp(smile_interp_df):

        X = np.array(smile_interp_df.index)
        Y = smile_interp_df.columns.to_numpy().astype(float) 
        Z = smile_interp_df.values.T  

        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
        fig.update_layout(
            title='IV Survace',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        fig.show()
