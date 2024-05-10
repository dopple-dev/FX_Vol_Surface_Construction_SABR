import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import scipy.optimize as opt
from scipy.stats import norm

from fx_vol_surf.Pricing.FX_Black_Scholes import Black_Scholes as BS

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



class build_smile:

    def ATM_K(T,vol,rd,rf,S):
        '''
        Find the ATM strike using the formulas in table 1.2 

        Input:
        - T: maturity 
        - vol: volatility 
        - rd, rf: risk free rates (domestic and foreign)
        - S: spot fx

        Output: 
        - ATM strike
        '''
        return np.exp(0.5*(vol**2)*T+(rd-rf)*T)*S

    def get_delta(sigma,K,T,w,S,rd,rf, delta_convention, delta_mrk=0):
        '''
        Find the delta of an option or a strategy (used in optimization function 'MS_K')

        Input: 
        - sigma: volatility
        - K: stike (guessed in 'MS_K')
        - T: tenor
        - w: 1.0 for call and -1.0 for put 
        - S: spot fx 
        - rd, rf: risk free rates (domestic and foreign)
        - delta_convention: types of delta convention used 

        Output: 
        - option delta 
        '''
        # find the Black-Scholes d1 
        d1 = (np.log(S / K) + (rd - rf + .5 * sigma * sigma) * T) / (sigma*np.sqrt(T))

        # inversion formula for retriving delta given all the input and d1 (based on the convention)
        if delta_convention == 'spot': 
            delta = w * np.exp(-rf * T) * norm.cdf(w * d1) # page 67 clark 
        elif delta_convention == 'fwd':
            delta = w * norm.cdf(w * d1)
            
        return delta

    def MS_K(delta_mrk,T,sigma,w,k0,S,rd,rf,delta_convention):
        '''
        Minimize the squared difference of the market delta with delta calculated guessing the option/strategy strike

        Input: 
        - delta_mrk: option delta (from market data) 
        - T: tenor
        - sigma: volatility
        - w: 1.0 for call and -1.0 for put 
        - k0: initial stike guess
        - S: spot fx 
        - rd, rf: risk free rates (domestic and foreign)
        - delta_convention: types of delta convention used 

        Output: 
        - option strike obtained through the minimization process 
        '''

        def optK_delta_diff(K):
            delta_opt = build_smile.get_delta(sigma,K,T,w,S,rd,rf, delta_convention, delta_mrk)
            return (delta_opt - delta_mrk)**2

        return fsolve(optK_delta_diff, k0)
    
    def SABR_vol(a, v, rho, K, T, b, S,rd,rf):

        F = S * np.exp((rd - rf) * T)
        z = v / a * (F * K) ** ((1 - b) / 2) * np.log(F / K)
        x = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
        vol = a / ((F * K) ** ((1 - b) / 2) * (
                1 + (1 - b) ** 2 / 24 * (np.log(F / K)) ** 2 + (1 - b) ** 4 / 1920 * (np.log(F / K)) ** 4)) * (z / x) * (
                        1 + ((1 - b) ** 2 / 24 * a ** 2 / (F * K) ** (1 - b) + 1 / 4 * rho * b * v * a / (F * K) ** (
                        (1 - b) / 2) + (2 - 3 * rho ** 2) / 24 * v ** 2) * T)
        return vol

    def fx_smile_constraints(atm, rr, ms, k_atm, ms_call_k, ms_put_k, T, Delta, S, rd, rf, delta_convention):
        '''
        Find SABR smile parameters given constraints imposed to match market quotes 

        Input: 
        - atm: market volatiltiy of atm options  
        - rr: market volatility of risk reversal strategies 
        - ms: market volatility of market strangle strategies 
        - k_atm: atm strike 
        - ms_call_k: call market strangle strike 
        - ms_put_k: put market strangle strike 
        - T: maturity 
        - Delta: option or strategy delta 
        - S: spot fx 
        - rd, rf: risk free rates (domestic and foreign) 
        - delta_convention: types of delta convention used

        Output:
        - SABR parameters and strike and volatilities for the entire market smile 
        '''

        def smile_conditions(x):

            '''
            x[0] is alpha,
            x[1] is nu
            x[2] is rho
            x[3] = strike call
            x[4] = strike put
            '''
            
            beta=1
            CALL = 1
            PUT = -1

            # Condition 1:  vol(ATM_strike) under SABR = atm market vol
            Condition_1_ATM = build_smile.SABR_vol(x[0], x[1], x[2], k_atm, T, beta, S,rd,rf) - atm

            # Condition 2: delta(RR_call) = delta(RR) = 0.25
            Condition_2_delta_RR_call = build_smile.get_delta(build_smile.SABR_vol(x[0], x[1], x[2], x[3], T, beta, S,rd,rf), x[3], T, CALL,S,rd,rf,delta_convention) - Delta

            # Condition 3: delta(RR_put) = -delta(RR) = -0.25
            # Condition_3_delta_RR_put = build_smile.get_delta(build_smile.SABR_vol(x[0], x[1], x[2], x[3], T, beta, S,rd,rf), x[4], T, PUT,S,rd,rf,delta_convention) + Delta
            Condition_3_delta_RR_put = build_smile.get_delta(build_smile.SABR_vol(x[0], x[1], x[2], x[4], T, beta, S,rd,rf), x[4], T, PUT,S,rd,rf,delta_convention) + Delta
    
            # Condition 4: vol(RR_call) - vol(RR_put) = vol(RR)
            Condition_4_RR = build_smile.SABR_vol(x[0], x[1], x[2], x[3], T, beta, S,rd,rf) - build_smile.SABR_vol(x[0], x[1], x[2], x[4], T, beta, S,rd,rf) - rr

            # Condition 5: vol(SS) = 0.5[ vol(call) + vol(put) ] - vol(atm)
            #Condition_5_SS = 0.5 * (SABR_vol(x[0], x[1], x[2], x[3], T, beta, S,rd,rf) + SABR_vol(x[0], x[1], x[2], x[4], T, beta, S,rd,rf)) - atm - ms
            
            # Condition 6: MS_price(MS_vol, K_MS_call) + MS_price(MS_vol, K_MS_put) = MS_price(call_vol, K_MS_call) + MS_price(put_vol, K_MS_put)
            Condition_6_MS_price = BS.BS_pricer(1,S,ms_call_k,T,ms+atm,rd,rf) - BS.BS_pricer(1,S,ms_call_k,T,build_smile.SABR_vol(x[0], x[1], x[2], x[3], T, beta, S,rd,rf),rd,rf) + BS.BS_pricer(-1,S,ms_put_k,T,ms+atm,rd,rf) - BS.BS_pricer(-1,S,ms_put_k,T,build_smile.SABR_vol(x[0], x[1], x[2], x[4], T, beta, S,rd,rf),rd,rf)

            return np.array([Condition_1_ATM, 
                            Condition_2_delta_RR_call, 
                            Condition_3_delta_RR_put, 
                            Condition_4_RR, 
                            #Condition_5_SS, 
                            Condition_6_MS_price
                            ]).flatten()
        
        return opt.least_squares(smile_conditions, x0=np.array([0.1, 0.999, -0.1, 1.4, 1.4]),
                                    bounds=([0, 0, -1, 0, 0], [np.inf, np.inf, 1, np.inf, np.inf])).x
    
    def construct_fx_smile(Delta,Expiry,ATM,MS,RR,S,rd,rf,delta_convention,beta_arr,tenor):
        
        # First step: find strikes for the ATM options and the MS strategies
        # ATM strke: see table 1.2
        K_ATM = build_smile.ATM_K(Expiry, ATM, rd,rf,S)

        # Market strangle strikes: see section 3.1.3 (volatility used is sigma_ATM + sigma_MS)
        K_MS_Call = np.array([build_smile.MS_K(Delta[i], Expiry[i], ATM[i] + MS[i], 1, S[i],S[i],rd[i],rf[i], delta_convention[i]) for i in range(len(S))]).flatten()
        K_MS_Put = np.array([build_smile.MS_K(-Delta[i], Expiry[i], ATM[i] + MS[i], -1, S[i],S[i],rd[i],rf[i], delta_convention[i]) for i in range(len(S))]).flatten()
        
        # find the SABR parameters and the strikes for all the market option 
        pars_strikes = np.array([build_smile.fx_smile_constraints(ATM[i], RR[i], MS[i], K_ATM[i], K_MS_Call[i], K_MS_Put[i], 
                                     Expiry[i], Delta[i], S[i], rd[i], rf[i], delta_convention[i]) for i in range(len(S))])

        # strore SABR parameters and strikes  
        alpha_set=[]
        nu_set=[]
        rho_set=[]
        K_call_set=[]
        K_Put_set=[]
        for i in pars_strikes:
            alpha_set.append(i[0])
            nu_set.append(i[1])
            rho_set.append(i[2])
            K_call_set.append(i[3])
            K_Put_set.append(i[4])
        
        # compute SABR volatility at the strikes obtained with calibration 
        vol_atm=np.array([build_smile.SABR_vol(alpha_set[i],nu_set[i],rho_set[i],K_ATM[i],Expiry[i],1,S[i],rd[i],rf[i]) for i in range(len(S))])
        vol_call=np.array([build_smile.SABR_vol(alpha_set[i],nu_set[i],rho_set[i],K_call_set[i],Expiry[i],1,S[i],rd[i],rf[i]) for i in range(len(S))])
        vol_put=np.array([build_smile.SABR_vol(alpha_set[i],nu_set[i],rho_set[i],K_Put_set[i],Expiry[i],1,S[i],rd[i],rf[i]) for i in range(len(S))])

        # dataframe containing al the data obtained 
        smile_SABR=pd.DataFrame({
                            "Tenor_str":tenor,
                            "Tenor_float":Expiry,
                            "Delta": Delta,
                            "Alpha":alpha_set,
                            "Beta":beta_arr,
                            "Vega":nu_set,
                            "Rho":rho_set,
                            "K_ATM":K_ATM,
                            "K_Call":K_call_set,
                            "K_Put":K_Put_set,
                            "Vol_ATM":vol_atm,
                            "Vol_Call":vol_call,
                            "Vol_Put":vol_put
                            })
        
        # check for precision of the smile construction scheme 

        # risk reversal condition 
        smile_SABR['mrk_rr_vol'] = RR
        smile_SABR['SABR_rr_vol'] = smile_SABR['Vol_Call'] - smile_SABR['Vol_Put']
        smile_SABR['rr_cond_error'] = smile_SABR['mrk_rr_vol'] - smile_SABR['SABR_rr_vol']

        # market strangle condiion 
        # smile_SABR['ms'] = np.array([BS.BS_pricer(1,S[i],K_MS_Call[i],Expiry[i],MS[i] + ATM[i],rd[i],rf[i]) for i in range(len(S))]) + np.array([BS.BS_pricer(-1,S[i],K_MS_Put[i],Expiry[i],MS[i] + ATM[i],rd[i],rf[i]) for i in range(len(S))])
        # smile_SABR['ms_t'] = np.array([BS.BS_pricer(1,S[i],K_MS_Call[i],Expiry[i],vol_call[i],rd[i],rf[i]) for i in range(len(S))]) + np.array([BS.BS_pricer(-1,S[i],K_MS_Put[i],Expiry[i],vol_put[i],rd[i],rf[i]) for i in range(len(S))])
        # smile_SABR['ms_test'] = smile_SABR['ms'] - smile_SABR['ms_t']

        # smile strangle condition 
        smile_SABR['ss_call'] = ATM + 0.5*RR - smile_SABR['Vol_Call']
        smile_SABR['ss_put'] = ATM - 0.5*RR - smile_SABR['Vol_Put']
        smile_SABR['ss_cond_error'] = smile_SABR['ss_call'] - smile_SABR['ss_put']

        return smile_SABR
